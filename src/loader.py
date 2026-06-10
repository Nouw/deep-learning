from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch

LABEL_MAP = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3,
}


class MEGDataset(Dataset):
    def __init__(
            self,
            files,
            window_seconds: float = 3.0,
            overlap: float = 0.5,
            sampling_rate: float = 2034 / 51,
    ):
        self.files = [Path(f) for f in files]
        self.sampling_rate = sampling_rate
        self.window_size = int(window_seconds * sampling_rate)
        self.stride = int(self.window_size * (1 - overlap))
        self.arrays = {}

        if self.stride <= 0:
            raise ValueError("overlap is too large; stride becomes 0.")

        self.items = []

        for path in sorted(self.files):
            name = path.stem
            label_str, subject_id, chunk_id = name.rsplit("_", 2)
            label = LABEL_MAP[label_str]

            arr = np.load(path, mmap_mode="r")
            self.arrays[path] = arr
            n_time = arr.shape[1]

            for start in range(0, n_time - self.window_size + 1, self.stride):
                end = start + self.window_size
                self.items.append({
                    "path": path,
                    "label": label,
                    "subject_id": subject_id,
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        arr = self.arrays[item["path"]]
        window = arr[:, item["start"]:item["end"]]
        x = torch.from_numpy(window.copy()).float()
        y = torch.tensor(item["label"], dtype=torch.long)

        return x, y
