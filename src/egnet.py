import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=248, n_classes=4, 
                 n_temporal_filters=16, kernel_length=128,
                 n_spatial_filters=2, n_separable_filters=32,
                 dropout=0.5):
        super(EEGNet, self).__init__()
        