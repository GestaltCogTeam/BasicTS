"""Adapted from blisky-li/APT (98a4c9c), baselines/Normalization/normalization/DAIN_denorm.py."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DAIN(nn.Module):
    """
    Paper: Deep Adaptive Input Normalization for Time Series Forecasting
    Link: https://doi.org/10.1109/TNNLS.2019.2944933
    Ref Official Code: https://github.com/passalis/dain
    Venue: IEEE Transactions on Neural Networks and Learning Systems 2020
    Task: Adaptive Input Normalization for Time Series Forecasting

    Note:
        The reversible denormalization interface follows the APT integration:
        https://github.com/blisky-li/APT/tree/main/baselines/Normalization/normalization
    """

    def __init__(self, mode='avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, **model_args):
        super().__init__()

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr
        self.input_dim = model_args['enc_in']
        # Parameters for adaptive average
        self.mean_layer = nn.Linear(self.input_dim,  self.input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye( self.input_dim,  self.input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(self.input_dim, self.input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(self.input_dim, self.input_dim)

        self.eps = 1e-8

    def _get_statistics(self, x):
        x = x.transpose(1, 2)
        if self.mode is None:
            pass
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            self.avg = avg.unsqueeze(-1)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            self.adaptive_avg = adaptive_avg.unsqueeze(-1)
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            self.adaptive_avg = adaptive_avg.unsqueeze(-1)
            x = x - self.adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = torch.where(adaptive_std <= self.eps, torch.ones_like(adaptive_std), adaptive_std)

            self.adaptive_std = adaptive_std.unsqueeze(-1)
        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            self.adaptive_avg = adaptive_avg.unsqueeze(-1)
            x = x - self.adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std = torch.where(adaptive_std <= self.eps, torch.ones_like(adaptive_std), adaptive_std)

            self.adaptive_std = adaptive_std.unsqueeze(-1)
            x = x / self.adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            self.gate = gate.unsqueeze(-1)


    def forward(self, x, normalize_mode:str):
        if normalize_mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif normalize_mode == 'denorm':
            x = self._denormalize(x)

        else: raise NotImplementedError

        return x

    def _normalize(self, x):
        x = x.transpose(1, 2)
        if self.mode == 'avg':
            x = x - self.avg

        elif self.mode == 'adaptive_avg':
            x = x - self.adaptive_avg

        elif self.mode == 'adaptive_scale':
            x = x - self.adaptive_avg
            x = x / (self.adaptive_std)

        elif self.mode == 'full':
            x = x - self.adaptive_avg
            x = x / (self.adaptive_std)
            x = x * self.gate

        x = x.transpose(1, 2)
        return x

    def _denormalize(self, x):
        x = x.transpose(1, 2)

        if self.mode == 'avg':
            x = x + self.avg

        elif self.mode == 'adaptive_avg':
            x = x + self.adaptive_avg

        elif self.mode == 'adaptive_scale':
            x = x * (self.adaptive_std)
            x = x + self.adaptive_avg

        elif self.mode == 'full':
            x = x / self.gate
            x = x * (self.adaptive_std)
            x = x + self.adaptive_avg

        x = x.transpose(1, 2)
        return x
