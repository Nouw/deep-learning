import torch
import matplotlib.pyplot as plt
import numpy as np

def predict(model, dataset, scaler, n_steps= 200, key="", plot_path = None):
    window = dataset.X[-1].clone().float()  # last known window, shape [window_size]
    predictions_scaled = []

    with torch.no_grad():
        for _ in range(n_steps):
            x = window.unsqueeze(0)               # [1, window_size]
            pred = model(x).squeeze().item()
            predictions_scaled.append(pred)
            window = torch.cat([window[1:], torch.tensor([pred])])

    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).squeeze()

    plt.figure(figsize=(12, 4))
    plt.plot(predictions, color="tomato", linewidth=1.2, label=f"{key} — {n_steps}-step forecast")
    plt.xlabel("Steps ahead")
    plt.ylabel("Value")
    plt.title(f"{key} - Recursive {n_steps}-step forecast")
    plt.legend()
    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path)

    plt.show()
