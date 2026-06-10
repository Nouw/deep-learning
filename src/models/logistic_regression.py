import torch
import torch.nn as nn


class LogisticRegressionMEG(nn.Module):
    """
    Simple linear baseline.

    Expected input shape:
        x: (batch, channels, time)

    Output:
        logits: (batch, num_classes)
    """

    def __init__(self, input_channels: int, input_time: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Linear(input_channels * input_time, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits