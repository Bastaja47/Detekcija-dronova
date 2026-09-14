"""
models.py

Deljene definicije arhitektura modela - koriste se u notebook-ovima
za trening/evaluaciju, i u api.py za serviranje.
"""

import torch.nn as nn


class ConfigurableCNN(nn.Module):
    """
    CNN arhitektura sa podesivim brojem conv slojeva, filtera i dropout-a.
    Ovo je arhitektura koju je Optuna optimizovala - koristimo je za finalni,
    servirani model (pokazala se bolja od BaselineCNN u evaluaciji).
    """

    def __init__(self, num_conv_layers=3, base_filters=16, dropout_rate=0.3, num_classes=7):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 2
        current_filters = base_filters
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=3, padding=1))
            in_channels = current_filters
            current_filters = current_filters * 2

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        final_spatial_size = 128 // (2 ** num_conv_layers)
        flattened_size = in_channels * final_spatial_size * final_spatial_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(self.relu(conv(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x