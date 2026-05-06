import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 20) → (batch, 1, 20): one timestep of 20 features
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
