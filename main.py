import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.cnn_lstm import HybridCNNLSTM
from src.models.lstm import LSTMNet
from src.models.net import Net
from src.models.recurrent_neural_network import RNNModel
from src.read_data import load_dataset, split_dataset
from src.test import test_model
from src.train import train_model
from src.predict import predict
import torch

logging.basicConfig(level=logging.DEBUG)

models = {
    "RNN": RNNModel(20, 64, 3),
    "Felix": Net(),
    "CNN-LSTM": HybridCNNLSTM(
        20,
        num_filters=64,
        kernel_size=5,
        hidden_size=256,
        num_layers=1,
        dropout=0.2,
    ),
    "LSTM": LSTMNet(
        hidden_size=256,
        num_layers=4,
        dropout=0.1,
    )
}

def train_mode():
    dataset, scaler = load_dataset('Xtrain.mat', 20)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_size=0.8, val_size=0.2, test_size=0.0)

    for key, model in models.items():
        folder = f"out/{key}/"

        if not os.path.exists(folder):
            os.mkdir(folder)

        best_model = train_model(model, train_dataset, val_dataset, epochs=100, model_name=key, plot_loss=True, plot_path=f"{folder}/loss_graph.png")

        torch.save(best_model, f"out/{key}/weights.pt")


def test_mode():
    dataset, scaler = load_dataset('Xtest.mat', 20, key="Xtest")
    test_dataset = DataLoader(dataset, batch_size = 32, shuffle=False)
    # _, _, test_dataset = split_dataset(dataset, train_size=0, val_size=0, test_size=1)

    for key, model in models.items():
        file_path = f"out/{key}/weights.pt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        models[key].load_state_dict(torch.load(file_path, weights_only=True))
        test_model(model, test_dataset, scaler, model_name=key, plot_path=f"out/{key}/test-predictions.png")

def predict_mode(n_steps=200):
    dataset, scaler = load_dataset('Xtrain.mat', 20)

    for key, model in models.items():
        file_path = f"out/{key}/weights.pt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        model.load_state_dict(torch.load(file_path, weights_only=True))
        model.eval()
        predict(model, dataset, scaler, n_steps, key, plot_path=f"out/{key}/future-predictions.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "predict"], help="Whether to train, test, or predict")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to predict (predict mode only)")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    elif args.mode == "test":
        test_mode()
    else:
        predict_mode(n_steps=args.steps)

if __name__ == "__main__":
    main()
