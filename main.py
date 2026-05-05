import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.models.cnn_lstm import HybridCNNLSTM
from src.models.net import Net
from src.read_data import load_dataset, split_dataset
from src.test import test_model
from src.train import train_model
from src.predict import predict
import torch

logging.basicConfig(level=logging.DEBUG)

def train_mode():
    dataset, scaler = load_dataset('Xtrain.mat', 20)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    models = {
        "Felix": Net(),
        "CNN-LSTM": HybridCNNLSTM(20, num_layers=2)
    }

    for key, model in models.items():
        folder = f"out/{key}/"

        if not os.path.exists(folder):
            os.mkdir(folder)

        best_model = train_model(model, train_dataset, val_dataset, plot_loss=True, plot_path=f"{folder}/loss_graph.png")

        torch.save(best_model, f"out/{key}/weights.pt")


def test_mode():
    models = {
        "Felix": Net(),
        "CNN-LSTM": HybridCNNLSTM(20, num_layers=2)
    }

    dataset, scaler = load_dataset('X', 20)
    train_dataset, _, test_dataset = split_dataset(dataset)

    for key, model in models.items():
        file_path = f"out/{key}/weights.pt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        models[key].load_state_dict(torch.load(file_path, weights_only=True))
        test_model(model, train_dataset, scaler, model_name=key, plot_path=f"out/{key}/test-predictions.png")

def predict_mode(n_steps=200):
    models = {
        "Felix": Net(),
        "CNN-LSTM": HybridCNNLSTM(20, num_layers=2)
    }

    dataset, scaler = load_dataset('Xtrain.mat', 20)

    for key, model in models.items():
        file_path = f"out/{key}/weights.pt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        model.load_state_dict(torch.load(file_path, weights_only=True))
        model.eval()
        predict(model, dataset, scaler, n_steps, key)

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
