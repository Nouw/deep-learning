import torch.nn as nn

class HybridCNNLSTM(nn.Module):
    def __init__(self,
                 window_size,
                 num_filters=128,
                 kernel_size=3,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.0
                 ):
        super(HybridCNNLSTM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out.squeeze(-1)
