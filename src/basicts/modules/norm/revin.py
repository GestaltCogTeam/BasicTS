from typing import Literal

import torch
from torch import nn


class RevIN(nn.Module):

    """
    RevIN
    Paper: Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift
    Official Code: https://github.com/ts-kim/RevIN
    Link: https://openreview.net/forum?id=cGDAkQo1C0p
    Venue: ICLR 2022
    """

    def __init__(self,
                 num_features: int = None,
                 eps: float = 1e-6,
                 affine: bool = True,
                 subtract_last: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param subtract_last: if True, subtract the last element of the time series instead of the mean
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: Literal["norm", "denorm"]) -> torch.Tensor:
        """
        :param x: input tensor of shape [batch_size, seq_len, num_features]
        :param mode: 'norm' for normalization, 'denorm' for denormalization
        :return: normalized or denormalized tensor of shape [batch_size, seq_len, num_features]
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise ValueError(f"Mode {mode} is not supported.")
        return x

    def _init_params(self):
        if self.num_features is None:
            raise ValueError("`num_features` need to be specified when `affine` = True.")
        # initialize RevIN params: [num_features,]
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
