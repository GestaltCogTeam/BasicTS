"""Adapted from blisky-li/APT (98a4c9c), FAN.py."""

import torch
from torch import nn


def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim = 1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, n=x.shape[1], dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered

    return norm_input, x_filtered



class FAN(nn.Module):
    """
    Paper: Frequency Adaptive Normalization For Non-stationary Time Series Forecasting
    Link: https://proceedings.neurips.cc/paper_files/paper/2024/file/37c6d0bc4d2917dcbea693b18504bd87-Paper-Conference.pdf
    Ref Official Code: https://github.com/wayne155/FAN
    Venue: NeurIPS 2024
    Task: Frequency Adaptive Normalization for Time Series Forecasting
    """
    def __init__(self, freq_topk = 20, rfft=True, **model_args):
        super().__init__()
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.enc_in = model_args['enc_in']
        self.epsilon = 1e-8
        self.freq_topk = freq_topk

        self.rfft = rfft

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):
        # freq normalization
        residual, pred_main  = main_freq_part(true, self.freq_topk, self.rfft)


        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)


    def normalize(self, inputs):
        # (B, T, N)
        bs, length, dim = inputs.shape
        norm_input, x_filtered = main_freq_part(inputs, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1,2), inputs.transpose(1,2)).transpose(1,2)

        return norm_input.reshape(bs, length, dim)


    def denormalize(self, input_norm):
        # inputs:  (B, O, N)
        # station_pred: outputs of normalize
        bs, length, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, length, dim)

    def forward(self, batch_x, mode='norm'):
        if mode == 'norm':
            return self.normalize(batch_x)
        elif mode =='denorm':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    """Predict future frequency components from filtered and raw histories."""

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )


    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
