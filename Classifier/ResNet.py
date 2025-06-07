import os
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class ResNetBlock(nn.Module): 
    def __init__(self, in_channel, channel_1, channel_2):
        super().__init__(); 
        self.conv_1 = nn.Conv2d(in_channel, channel_1, kernel_size = 3, padding = 1); 
        nn.init.kaiming_normal(self.conv_1.weight); 
        self.conv_2 = nn.Conv2d(channel_1, channel_2, kernel_size = 3, padding = 1); 
        nn.init.kaiming_normal(self.conv_2.weight); 

        self.batch_normalization_1 = nn.BatchNorm2d(channel_1); 
        self.batch_normalization_2 = nn.BatchNorm2d(channel_2); 

        self.relu = nn.ReLU(inplace = True); 
        self.conv_skip = nn.Conv2d(in_channel, channel_2, kernel_size = 1); 

    def forward(self, x): 
        skip = x; 
        skip = self.conv_skip(skip); 

        x = self.conv_1(x); 
        x = self.batch_normalization_1(x); 
        x = self.relu(x); 

        x = self.conv_2(x); 
        x = self.batch_normalization_2(x); 
        x = self.relu(x); 

        x += skip; # Residual skip

        return x; 


class ResidualNetwork(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, channel_4, channel_5, number_classes): 
        super().__init__(); 
        self.conv_1 = nn.Conv2d(in_channel, channel_1, kernel_size = 7, stride = 2); 
        self.batch_normalization_1 = nn.BatchNorm2d(channel_1); 
        nn.init.kaiming_normal(self.conv_1.weight); 

        self.conv_2x = ResNetBlock(channel_1, channel_2, channel_2); 
        self.conv_3x = ResNetBlock(channel_2, channel_3, channel_3); 
        self.conv_4x = ResNetBlock(channel_3, channel_4, channel_4); 
        self.conv_5x = ResNetBlock(channel_4, channel_5, channel_5); 

        self.fully_connected = nn.Linear(channel_5, number_classes); 

        self.relu = nn.ReLU(inplace = True); 
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2); 
        self.avgpool = nn.AdaptivePool2d(1); 

    def forward(self, x): 
        x = self.conv_1(x); 
        x = self.batch_normalization_1(x); 
        x = self.relu(x); 
        x = self.maxpool(x); 

        x = self.conv_2x(x);  
        x = self.conv_3x(x); 
        x = self.conv_4x(x); 
        x = self.conv_5x(x); 

        x = self.avgpool(x); 
        x = torch.flatten(x); 
        x = self.fully_connected(x); 

        return x; 

