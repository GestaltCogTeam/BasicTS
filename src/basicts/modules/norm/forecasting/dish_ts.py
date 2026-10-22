"""Adapted from blisky-li/APT (98a4c9c), baselines/Normalization/normalization/DishTS.py."""

import torch
import torch.nn.functional as F
from torch import nn


class DishTS(nn.Module):
    """
    Paper: Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting
    Link: https://ojs.aaai.org/index.php/AAAI/article/view/25914
    Ref Official Code: https://github.com/weifantt/Dish-TS
    Venue: AAAI 2023
    Task: Normalization and Denormalization for Time Series Forecasting
    """

    def __init__(self, init='avg',**model_args):
        super().__init__()
        # init = args.dish_init #'standard', 'avg' or 'uniform'
        self.init = init
        activate = True
        self.n_series = model_args['enc_in'] # number of series
        self.lookback = model_args['seq_len'] # lookback length
        if self.init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(self.n_series, self.lookback, 2)/self.lookback)
        elif self.init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(self.n_series, self.lookback, 2)/self.lookback)
        elif self.init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(self.n_series, self.lookback, 2)/self.lookback+torch.rand(self.n_series, self.lookback, 2)/self.lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(self.n_series)), nn.Parameter(torch.zeros(self.n_series))
        self.activate = activate

    def forward(self, batch_x, mode='norm', dec_inp=None):
        if mode == 'norm':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x #,dec_inp
        elif mode == 'denorm':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:]
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih
