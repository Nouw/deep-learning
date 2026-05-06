import logging
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_model(model: torch.nn.Module, test_loader, scaler, model_name="", plot_path=None):
    model.eval()
    all_predictions_scaled = np.array([])
    all_truth_scaled = np.array([])

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            all_predictions_scaled = np.concatenate([all_predictions_scaled, predictions.numpy().flatten()])
            all_truth_scaled = np.concatenate([all_truth_scaled, y_batch.numpy().flatten()])

    predictions_original = scaler.inverse_transform(all_predictions_scaled.reshape(-1, 1)).squeeze()
    targets_original = scaler.inverse_transform(all_truth_scaled.reshape(-1, 1)).squeeze()

    mae = mean_absolute_error(targets_original, predictions_original)
    mse = mean_squared_error(targets_original, predictions_original)

    logging.info(f"[{model_name}] Mean Absolute Error: {mae}")
    logging.info(f"[{model_name}] Mean Squared Error: {mse}")

    time_steps = np.arange(len(predictions_original))
    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, targets_original, label="Truth", color="steelblue")
    plt.plot(time_steps, predictions_original, label="Predictions", color="orange")
    plt.title(model_name)
    plt.legend()
    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path)

    plt.show()



def plot_predictions(model, data_scaled, data_original, scaler, window_size, model_name="Model"):
    all_predictions_scaled = []

    with torch.no_grad():
        for i in range(len(data_scaled) - window_size):
            window = data_scaled[i:i+window_size]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            prediction = model(x)
            all_predictions_scaled.append(prediction.item())

    predictions_original = scaler.inverse_transform(np.array(all_predictions_scaled).reshape(-1, 1)).squeeze()
    targets_original = data_original[window_size:]

    absolute_errors = np.abs(predictions_original - targets_original)
    time_steps      = np.arange(len(predictions_original))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(time_steps, targets_original,
             color='steelblue', linewidth=1.2,
             label='Real test values', zorder=3)

    ax1.plot(time_steps, predictions_original,
             color='tomato', linewidth=1.2, linestyle='--',
             label=f'{model_name} predictions', zorder=4)

    plt.show()
