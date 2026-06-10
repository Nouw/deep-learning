from pathlib import Path
import h5py
import numpy as np
import scipy.signal


def preprocess_h5_file(
    input_path: Path,
    output_path: Path,
    downsample_factor: int,
    eps: float,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing file: {output_path}")
        return

    with h5py.File(input_path, "r") as f:
        key = list(f.keys())[0]
        data = f[key][()]  # expected shape: (248, 35624)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array in {input_path}, got shape {data.shape}")

    decimated = scipy.signal.decimate(
        data,
        downsample_factor,
        axis=1,
        zero_phase=True,
    ).astype(np.float32)

    mean = decimated.mean(axis=1, keepdims=True)
    std = decimated.std(axis=1, keepdims=True)

    data_norm = (decimated - mean) / (std + eps)
    data_norm = data_norm.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data_norm)

    print(f"Saved {output_path} | shape={data_norm.shape}")


def preprocess_folder(
    input_folder: Path,
    output_folder: Path,
    downsample_factor: int,
    eps: float,
    force: bool,
) -> None:
    input_folder = input_folder.resolve()
    output_folder = output_folder.resolve()

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    h5_files = sorted(input_folder.glob("*.h5"))

    if not h5_files:
        print(f"No .h5 files found in {input_folder}")
        return

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(h5_files)} h5 files")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Effective sampling rate: {2034 / downsample_factor:.2f} Hz")
    print(f"Epsilon: {eps}")
    print()

    for input_path in h5_files:
        output_path = output_folder / f"{input_path.stem}.npy"

        preprocess_h5_file(
            input_path=input_path,
            output_path=output_path,
            downsample_factor=downsample_factor,
            eps=eps,
            force=force,
        )