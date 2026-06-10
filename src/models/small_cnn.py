import torch
from torch import nn

class SmallMEGCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.classifier(x)