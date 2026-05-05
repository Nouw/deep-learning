import scipy.io
import logging
from sklearn.preprocessing import MinMaxScaler
from src.dataset import LaserDataset
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

def load_dataset(filename: str, window_size: int):
    logging.debug(f'Loading {filename}')

    mat = scipy.io.loadmat(filename)
    data = mat['Xtrain'].squeeze()

    logging.debug(f'Loaded {filename}')
    logging.debug(f"Total number of data points: {len(data)}")
    logging.debug(f"Min value: {data.min():.4f}, Max value: {data.max():.4f}")

    plt.figure(figsize=(14, 4))
    plt.plot(data, linewidth=0.8, color='steelblue')
    plt.title("Training Data")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    scaled_data, scaler = scale_data(data)

    dataset = LaserDataset(scaled_data, window_size)

    return dataset, scaler

def scale_data(data):
    data_2d = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_2d).squeeze()

    return data_scaled, scaler

def split_dataset(
        data: LaserDataset,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        batch_size = 32
):
    length = len(data)
    train_data_size = int(train_size * length)
    val_data_size = int(val_size * length)
    test_data_size = int(test_size * length)

    train_dataset = Subset(data, range(0, train_data_size))
    val_dataset = Subset(data, range(train_data_size, train_data_size + val_data_size))
    test_dataset = Subset(data, range(train_data_size + val_data_size, train_data_size + val_data_size + test_data_size))

    logging.debug(f"Training samples: {len(train_dataset)}")
    logging.debug(f"Validation samples: {len(val_dataset)}")
    logging.debug(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

