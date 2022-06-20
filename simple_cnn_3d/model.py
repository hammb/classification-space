import torch
from torch import nn
from torch.nn import functional as F


class SimpleCnn3d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        conv_pool_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2, 2)
        )

    def forward(self, x):
        y = self.conv_pool_layer()
