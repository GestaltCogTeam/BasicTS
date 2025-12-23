# pylint: disable=not-callable
from typing import Callable, Dict

import torch
from torch import nn

from basicts.metrics import ALL_METRICS
from basicts.modules.norm import RevIN

from ..config.fits_config import FITSConfig


class FITS(nn.Module):

    """
    Paper: FITS: Modeling Time Series with 10k parameters

    Official Code: https://github.com/VEWOXIC/FITS

    Link: https://arxiv.org/abs/2307.03756

    Venue: ICLR 2024
    
    Task: Time Series Forecasting
    """

    def __init__(self, config: FITSConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.individual = config.individual
        self.cut_freq = config.cut_freq
        if self.cut_freq is None:
            self.cut_freq = (self.input_len // config.base_period + 1) * config.h_order + 10
        self.len_ratio = (self.input_len + self.output_len) / self.input_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.cut_freq, int(self.cut_freq * self.len_ratio)).to(torch.cfloat)
                for _ in range(config.num_features)])
        else:
            # complex layer for frequency upsampling
            self.freq_upsampler = nn.Linear(
                self.cut_freq, int(self.cut_freq * self.len_ratio)).to(torch.cfloat)

        self.revin = RevIN(affine=False)

        # loss function
        if isinstance(config.loss, Callable):
            self.loss_fn = config.loss
        elif isinstance(config.loss, str) and config.loss in ALL_METRICS:
            self.loss_fn = ALL_METRICS[config.loss]
        else:
            raise ValueError(f"Loss {config.loss} is not supported.")

        self.training_mode = config.training_mode
        assert self.training_mode in ["xy", "y"], \
            f"Training mode {self.training_mode} is not supported."

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of FITS.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].
            targets (torch.Tensor): Target tensor of shape [batch_size, output_len, num_features].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the prediction tensor of shape 
            [batch_size, output_len, num_features] and the loss tensor.
        """

        x = self.revin(inputs, "norm")

        # low pass filter
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.cut_freq:] = 0
        low_specx = low_specx[:, 0:self.cut_freq, :]

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0),int(self.cut_freq * self.len_ratio), low_specx.size(2)],
                 dtype=low_specx.dtype).to(low_specx.device)
            for i, layer in enumerate(self.freq_upsampler):
                low_specxy_[:, :, i]=layer(low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        low_specxy = torch.zeros(
            [low_specxy_.size(0),int((self.input_len + self.output_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:] = low_specxy_ # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.len_ratio # energy compemsation for the length change

        prediction = self.revin(low_xy, "denorm")

        if self.training_mode == "xy":
            targets = torch.cat([inputs, targets], dim=1)
            loss = self.loss_fn(prediction, targets)
        else:
            loss = self.loss_fn(prediction[:, -self.output_len:, :], targets)
        return {"prediction": prediction[:, -self.output_len:, :], "loss": loss}
