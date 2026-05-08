import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, filters = 4, kernel_size = 5, dropout_chance=0.1):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1,filters,kernel_size,padding=2)
        self.conv2 = nn.Conv1d(filters,filters,kernel_size,padding=2)
        self.conv3 = nn.Conv1d(filters,filters,kernel_size,padding=2)
        self.conv4 = nn.Conv1d(filters,filters,kernel_size,padding=2)

        self.bn1 = nn.BatchNorm1d(filters)
        self.bn2 = nn.BatchNorm1d(filters)
        self.bn3 = nn.BatchNorm1d(filters)
        self.bn4 = nn.BatchNorm1d(filters)

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_chance)
        self.fc = nn.Linear(filters,1)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(F.relu(self.bn4(self.conv4(x))))

        #x = self.globalpool(x)
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x

