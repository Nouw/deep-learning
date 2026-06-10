import json
from itertools import product
from pathlib import Path
import time
import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from src.loader import LABEL_MAP, MEGDataset
from src.models.cnn import CNN
from src.models.logistic_regression import LogisticRegressionMEG
from src.models.small_cnn import SmallMEGCNN

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    if minutes:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_default_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError("No GPU found. This training script expects CUDA or MPS.")


def prepare_device(device):
    device = get_default_device() if device is None else torch.device(device)

    if device.type == "cpu":
        raise RuntimeError("CPU training is disabled. Use a CUDA or MPS device.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    return device


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def expand_grid(config):
    keys = list(config.keys())
    values = [
        value if isinstance(value, list) else [value]
        for value in config.values()
    ]

    for combination in product(*values):
        yield dict(zip(keys, combination))


def format_csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def write_epoch_history(history, history_csv):
    history_csv = Path(history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    for column in history_df.columns:
        history_df[column] = history_df[column].apply(format_csv_value)
    history_df.to_csv(history_csv, index=False)


def safe_filename_part(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"


def build_model(model_name, input_channels, input_time, num_classes, model_params):
    if model_name == "cnn":
        return CNN(
            input_channels=input_channels,
            num_classes=num_classes,
            **model_params,
        )

    if model_name == "logistic_regression":
        return LogisticRegressionMEG(
            input_channels=input_channels,
            input_time=input_time,
            num_classes=num_classes,
        )

    if model_name == "small_cnn":
        return SmallMEGCNN(n_channels=input_channels, n_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def move_batch(x, y, device):
    non_blocking = device.type == "cuda"
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


def run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x, y in loader:
        x, y = move_batch(x, y, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_examples += x.size(0)

    return total_loss / total_examples


def evaluate(model, loader, criterion, device, num_classes, use_amp):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = move_batch(x, y, device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_examples += x.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "val_loss": total_loss / total_examples,
        "val_accuracy": accuracy_score(targets, predictions),
        "val_macro_f1": f1_score(targets, predictions, labels=labels, average="macro", zero_division=0),
        "val_weighted_f1": f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }

    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"{class_name}_precision"] = precision[index]
        metrics[f"{class_name}_recall"] = recall[index]
        metrics[f"{class_name}_f1"] = f1[index]

    return metrics


def train_one_config(
    train_files,
    val_files,
    model_name,
    model_params,
    training_params,
    device=None,
    run_index=None,
    history_csv=None,
):
    device = prepare_device(device)
    window_seconds = training_params.get("window_seconds", 3)
    overlap = training_params.get("overlap", 0.5)
    batch_size = training_params.get("batch_size", 32)
    num_workers = training_params.get("num_workers", 2)
    epochs = training_params.get("epochs", 10)
    patience = training_params.get("patience", epochs)
    learning_rate = training_params.get("learning_rate", 3e-4)
    weight_decay = training_params.get("weight_decay", 0.0)
    num_classes = training_params.get("num_classes", len(LABEL_MAP))

    train_dataset = MEGDataset(train_files, window_seconds=window_seconds, overlap=overlap)
    val_dataset = MEGDataset(val_files, window_seconds=window_seconds, overlap=overlap)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation datasets must both contain at least one window.")

    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, device=device)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, device=device)

    sample_x, _ = train_dataset[0]
    model = build_model(
        model_name=model_name,
        input_channels=sample_x.shape[0],
        input_time=sample_x.shape[1],
        num_classes=num_classes,
        model_params=model_params,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_amp = training_params.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metrics = None
    best_score = -np.inf
    epochs_without_improvement = 0
    run_start = time.perf_counter()
    best_epoch_seconds = None
    epoch_history = []

    for epoch in range(1, epochs + 1):
        synchronize_device(device)
        epoch_start = time.perf_counter()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        metrics = evaluate(model, val_loader, criterion, device, num_classes, use_amp)

        synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start

        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch
        metrics["epoch_seconds"] = epoch_seconds
        epoch_history.append({
            "run_index": run_index,
            "model": model_name,
            **model_params,
            **training_params,
            **metrics,
        })

        if metrics["val_macro_f1"] > best_score:
            best_score = metrics["val_macro_f1"]
            best_metrics = metrics
            best_epoch_seconds = epoch_seconds
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"val_macro_f1={metrics['val_macro_f1']:.4f} | "
            f"time={format_seconds(epoch_seconds)}"
        )

        if epochs_without_improvement >= patience:
            print(f"  stopping early after {epoch} epochs")
            break

    synchronize_device(device)
    total_seconds = time.perf_counter() - run_start
    if history_csv is not None:
        write_epoch_history(epoch_history, history_csv)

    result = {
        "run_index": run_index,
        "model": model_name,
        "device": str(device),
        "best_epoch": best_metrics["epoch"],
        "best_epoch_seconds": best_epoch_seconds,
        "total_seconds": total_seconds,
        "number_of_parameters": count_parameters(model),
        **model_params,
        **training_params,
        **best_metrics,
    }
    if history_csv is not None:
        result["history_csv"] = str(history_csv)
    result["confusion_matrix"] = json.dumps(result["confusion_matrix"])
    return result


def run_experiments(
    train_files,
    val_files,
    experiment_configs,
    output_csv="../results/intra_results.csv",
    device=None,
    save_run_history=False,
    history_dir=None,
):
    rows = []
    device = prepare_device(device)
    output_csv = Path(output_csv)
    if history_dir is None:
        history_dir = output_csv.parent / f"{output_csv.stem}_history"
    history_dir = Path(history_dir)

    all_runs = [
        (experiment["model"], model_params, training_params)
        for experiment in experiment_configs
        for model_params in expand_grid(experiment.get("model_params", {}))
        for training_params in expand_grid(experiment.get("training_params", {}))
    ]
    total_start = time.perf_counter()

    for run_index, (model_name, model_params, training_params) in enumerate(all_runs, start=1):
        print(
            f"\nRun {run_index}/{len(all_runs)} | "
            f"model={model_name} | model_params={model_params} | training_params={training_params}"
        )
        history_csv = None
        if save_run_history:
            history_csv = history_dir / f"run_{run_index:03d}_{safe_filename_part(model_name)}.csv"

        rows.append(
            train_one_config(
                train_files=train_files,
                val_files=val_files,
                model_name=model_name,
                model_params=model_params,
                training_params=training_params,
                device=device,
                run_index=run_index,
                history_csv=history_csv,
            )
        )
        print(f"Finished run {run_index}/{len(all_runs)} in {format_seconds(rows[-1]['total_seconds'])}")

    results = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"\nFinished {len(all_runs)} runs in {format_seconds(time.perf_counter() - total_start)}")
    print(f"Saved results to {output_csv}")
    return results
