#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2022/01/31 20:45:33
@Author  :   Bo 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_default_resnet_model(wavenumber, n_class, quantification=False, detection=True):
    layers = 6
    hidden_size = 100
    block_size = 2
    hidden_sizes = [hidden_size] * layers
    num_blocks = [block_size] * layers
    in_channels = 64
    resnet_model = ResNet(hidden_sizes=hidden_sizes, num_blocks=num_blocks, input_dim=wavenumber, 
                          in_channels=in_channels, n_classes=n_class, quantification=quantification, detection=detection)
    return resnet_model


class ResNet(nn.Module):
    def __init__(self, hidden_sizes, num_blocks, input_dim=1000,
        in_channels=64, n_classes=30, quantification=False, detection=True):
        super(ResNet, self).__init__()
        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.quantification=quantification
        self.detection = detection 
        
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
            padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()
        if self.detection:
            self.linear = nn.Linear(self.z_dim, self.n_classes)
        if self.quantification:
            self.quan_linear = nn.Linear(self.z_dim, 1)


    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        z = self.encode(x)
        if self.detection:
            pred = self.linear(z)
        else:
            pred = []
        if self.quantification:
            quan = self.quan_linear(z)[:, 0]
        else:
            quan = []
        return z, pred, quan
    
    def forward_test(self, x):
        z = self.encode(x)
        if self.detection:
            pred = self.linear(z)
        else:
            pred = []
        if self.quantification:
            quan = self.quan_linear(z)[:, 0]
        else:
            quan = []
        return z, pred, quan 

    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim


def add_activation(activation='relu'):
    """
    Adds specified activation layer, choices include:
    - 'relu'
    - 'elu' (alpha)
    - 'selu'
    - 'leaky relu' (negative_slope)
    - 'sigmoid'
    - 'tanh'
    - 'softplus' (beta, threshold)
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    # SOFTPLUS DOESN'T WORK with automatic differentiation in pytorch
    elif activation == 'softplus':
        return nn.Softplus(beta=1, threshold=20)