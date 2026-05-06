import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, #alleen laserwaarde op 1 punt
            hidden_size=hidden_size, #vector van 64 getallen voor onthouden van complexe patronen
            num_layers=num_layers, #tweelaags LSTM
            dropout=dropout, #tegen overfitting, 10 procent dropout tussen lagen
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq_len) → (batch, seq_len, 1)

        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]  # shape: (batch, hidden_size)

        out = self.fc(last_hidden)  # shape: (batch, 1)
        return out
