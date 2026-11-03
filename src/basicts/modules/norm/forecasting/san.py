"""Adapted from blisky-li/APT (98a4c9c), SAN.py."""

import torch
from torch import nn


class SAN(nn.Module):
    """
    Paper: Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective
    Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html
    Ref Official Code: https://github.com/ustc-time-series/SAN
    Venue: NeurIPS 2023
    Task: Slice-level Normalization for Time Series Forecasting
    """

    def __init__(self, features='M',station_type='adaptive', **model_args):
        super().__init__()
        self.configs = model_args
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.period_len = model_args['period_len']
        self.features = features
        self.channels = model_args['enc_in'] if self.features == 'M' else 1
        self.station_type = station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):

        channels = self.channels
        seq_len = self.seq_len // self.period_len
        pred_len = self.pred_len_new
        period_len = self.period_len
        self.model = MLP(mode='mean', seq_len=seq_len, pred_len=pred_len, channels=channels, period_len=period_len).float()
        self.model_std = MLP(mode='std',  seq_len=seq_len, pred_len=pred_len, channels=channels, period_len=period_len).float()

    def normalize(self, inputs):
        if self.station_type == 'adaptive':
            bs, length, dim = inputs.shape
            inputs = inputs.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(inputs, dim=-2, keepdim=True)
            std = torch.std(inputs, dim=-2, keepdim=True)
            norm_input = (inputs - mean) / (std + self.epsilon)
            inputs = inputs.reshape(bs, length, dim)
            mean_all = torch.mean(inputs, dim=1, keepdim=True)
            outputs_mean = self.model(mean.squeeze(2) - mean_all, inputs - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), inputs)
            # print(inputs.shape, mean.shape, outputs_mean.shape)
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)[:, -self.pred_len_new:, :]

            if self.features == 'MS':
                outputs = outputs[:, :, [self.channels - 1, -1]]
            # print(outputs.shape)
            self.station_pred = outputs

            return norm_input.reshape(bs, length, dim)

        else:
            return inputs, None

    def de_normalize(self, inputs):
        if self.station_type == 'adaptive':
            station_pred = self.station_pred
            bs, length, dim = inputs.shape
            inputs = inputs.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = inputs * (std + self.epsilon) + mean
            return output.reshape(bs, length, dim)
        else:
            return inputs
    def forward(self, inputs, mode='norm'):
        if mode == 'norm':
            return self.normalize(inputs)
        elif mode == 'denorm':
            return self.de_normalize(inputs)
        else:
            return inputs

class MLP(nn.Module):
    """Predict slice statistics from historical statistics and raw values."""

    def __init__(self, mode, seq_len, pred_len, channels, period_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_len = period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
