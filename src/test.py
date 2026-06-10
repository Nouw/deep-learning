import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

from src.loader import LABEL_MAP, MEGDataset
from src.train import (
    CLASS_NAMES,
    build_model,
    count_parameters,
    format_seconds,
    make_loader,
    move_batch,
    prepare_device,
    synchronize_device,
)
from src.utilities import list_npy_files


__test__ = False


def make_dataset(files, training_params):
    return MEGDataset(
        files,
        window_seconds=training_params.get("window_seconds", 3),
        overlap=training_params.get("overlap", 0.5),
    )


def load_checkpoint(checkpoint_path):
    return torch.load(Path(checkpoint_path), map_location="cpu")


def load_trained_model(checkpoint_path, sample_x, device):
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]

    model = build_model(
        model_name=metadata["model_name"],
        input_channels=metadata.get("input_channels", sample_x.shape[0]),
        input_time=metadata.get("input_time", sample_x.shape[1]),
        num_classes=metadata.get("num_classes", len(LABEL_MAP)),
        model_params=metadata.get("model_params", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, metadata


def evaluate_loader(model, loader, criterion, device, num_classes, use_amp):
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
        "test_loss": total_loss / total_examples,
        "test_accuracy": accuracy_score(targets, predictions),
        "test_macro_f1": f1_score(targets, predictions, labels=labels, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }

    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"{class_name}_precision"] = precision[index]
        metrics[f"{class_name}_recall"] = recall[index]
        metrics[f"{class_name}_f1"] = f1[index]

    return metrics


def test_model_on_folders(
    checkpoint_path,
    test_folders,
    output_csv="../results/test_results.csv",
    batch_size=None,
    num_workers=None,
    device=None,
    include_combined=True,
):
    device = prepare_device(device)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    training_params = metadata.get("training_params", {})

    batch_size = batch_size or training_params.get("batch_size", 128)
    num_workers = training_params.get("num_workers", 0) if num_workers is None else num_workers
    num_classes = metadata.get("num_classes", len(LABEL_MAP))
    use_amp = training_params.get("use_amp", True) and device.type == "cuda"
    criterion = nn.CrossEntropyLoss()

    if isinstance(test_folders, (str, Path)):
        test_folders = [test_folders]

    test_folders = [Path(folder) for folder in test_folders]
    folder_files = {folder: list_npy_files(folder) for folder in test_folders}
    missing = [str(folder) for folder, files in folder_files.items() if not files]
    if missing:
        raise FileNotFoundError(f"No .npy files found in: {missing}")

    first_files = next(iter(folder_files.values()))
    first_dataset = make_dataset(first_files, training_params)
    sample_x, _ = first_dataset[0]
    model, metadata = load_trained_model(checkpoint_path, sample_x, device)

    rows = []
    all_files = []
    start = time.perf_counter()

    for folder, files in folder_files.items():
        all_files.extend(files)
        dataset = make_dataset(files, training_params)
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)

        synchronize_device(device)
        folder_start = time.perf_counter()
        metrics = evaluate_loader(model, loader, criterion, device, num_classes, use_amp)
        synchronize_device(device)

        rows.append({
            "split": folder.name,
            "folder": str(folder),
            "num_files": len(files),
            "num_windows": len(dataset),
            "model": metadata.get("model_name"),
            "device": str(device),
            "number_of_parameters": count_parameters(model),
            "seconds": time.perf_counter() - folder_start,
            **metadata.get("model_params", {}),
            **training_params,
            **metrics,
        })
        print(
            f"Evaluated {folder} | "
            f"windows={len(dataset)} | "
            f"accuracy={metrics['test_accuracy']:.4f} | "
            f"macro_f1={metrics['test_macro_f1']:.4f} | "
            f"time={format_seconds(rows[-1]['seconds'])}"
        )

    if include_combined and len(test_folders) > 1:
        dataset = make_dataset(all_files, training_params)
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)

        synchronize_device(device)
        combined_start = time.perf_counter()
        metrics = evaluate_loader(model, loader, criterion, device, num_classes, use_amp)
        synchronize_device(device)

        rows.append({
            "split": "combined",
            "folder": json.dumps([str(folder) for folder in test_folders]),
            "num_files": len(all_files),
            "num_windows": len(dataset),
            "model": metadata.get("model_name"),
            "device": str(device),
            "number_of_parameters": count_parameters(model),
            "seconds": time.perf_counter() - combined_start,
            **metadata.get("model_params", {}),
            **training_params,
            **metrics,
        })
        print(
            f"Evaluated combined test folders | "
            f"windows={len(dataset)} | "
            f"accuracy={metrics['test_accuracy']:.4f} | "
            f"macro_f1={metrics['test_macro_f1']:.4f} | "
            f"time={format_seconds(rows[-1]['seconds'])}"
        )

    results = pd.DataFrame(rows)
    results["confusion_matrix"] = results["confusion_matrix"].apply(json.dumps)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print(f"Saved test results to {output_csv}")
    print(f"Finished testing in {format_seconds(time.perf_counter() - start)}")
    return results
