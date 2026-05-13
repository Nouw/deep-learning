import h5py
import torch
from torch.utils.data import Dataset
import os
import scipy

class MEGDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        label_map = {"rest": 0, "task_story_math": 1, "task_working_memory": 2, "task_motor": 3 }

        for name in os.listdir(folder):
            if name.endswith("h5"):
                label_str = name.rsplit("_", 2)[0]
                self.samples.append((os.path.join(folder, name), label_map[label_str]))

    def __len__(self):
        return len(self.samples)

    # Niks boven de 100hz hebben
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            data = f[key][()]

        down_sampled = scipy.signal.decimate(data, 20)
        return torch.tensor(down_sampled.copy(), dtype=torch.float32), label
