import os
from pathlib import Path
import torch

def list_h5_files(folder):
    return [
        os.path.join(str(folder), name)
        for name in sorted(os.listdir(folder))
        if name.endswith(".h5")
    ]

def list_npy_files(folder):
    return [
        os.path.join(str(folder), name)
        for name in sorted(os.listdir(folder))
        if name.endswith(".npy")
    ]

def split_by_validation_chunks(all_files, val_chunks=("7", "8")):
    val_chunks = {str(chunk) for chunk in val_chunks}
    train_files = []
    val_files = []

    for file in all_files:
        chunk_id = Path(file).stem.rsplit("_", 1)[-1]

        if chunk_id in val_chunks:
            val_files.append(file)
        else:
            train_files.append(file)

    return train_files, val_files

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No GPU found. Training expects CUDA or MPS.")
