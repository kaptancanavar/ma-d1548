__author__ = "Semih Tarik Uenal"

import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, in_planes, planes):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_planes, planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        x = self.conv3(x)
        return self.relu(x + y)

class ResNet3D(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(ResNet3D, self).__init__()
        self.num_filters = num_filters
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock3D(in_channels, self.num_filters)
        self.resblock2 = ResBlock3D(self.num_filters, self.num_filters)
        self.resblock3 = ResBlock3D(self.num_filters, self.num_filters)
        self.resblock4 = ResBlock3D(self.num_filters, self.num_filters)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(self.num_filters, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.resblock1(y)
        y = self.resblock2(y)
        y = self.resblock3(y)
        y = self.resblock4(y)
        y = self.conv2(y)
        return x + y
