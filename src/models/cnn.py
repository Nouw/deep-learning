import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_classes: int,
            base_channels: int = 32,
            kernel_size: int = 7,
            dropout: float = 0.3,
            channels: tuple[int, int, int] | None = None,

    ):
        super().__init__()
        if channels is None:
            channels = (base_channels, base_channels * 2, base_channels * 4)

        c1, c2, c3 = channels
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=c1,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=kernel_size,
                padding=padding,

            ),

            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=c2,
                out_channels=c3,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits
