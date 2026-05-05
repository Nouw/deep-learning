from torch.utils.data import Dataset, DataLoader
import torch

class LaserDataset(Dataset):
    def __init__(self, series, window_size):
        self.window_size = window_size
        series_tensor = torch.tensor(series, dtype=torch.float32)

        X = []
        y = []

        for i in range(len(series_tensor) - window_size):
            X.append(series_tensor[i:i+window_size])
            y.append(series_tensor[i+window_size])

            self.X = torch.stack(X)
            self.y = torch.stack(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
