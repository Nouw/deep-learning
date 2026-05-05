import logging
import copy
import os.path

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_model(
        model,
        train_loader,
        val_loader,
        epochs=100,
        learning_rate=0.001,
        plot_loss=False,
        plot_path=None
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_model_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        avg_train_loss = np.mean(batch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                batch_val_losses.append(loss.item())

        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch + 1} / {epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if plot_loss:
        plt.figure(figsize=(12, 5))
        plt.plot(val_losses, label=f"(best: {best_val_loss:.5f})")

        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE Loss")
        plt.title("Model Comparison — Validation Loss Over Training")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if plot_path:
            plt.savefig(plot_path)

    return best_model_state
