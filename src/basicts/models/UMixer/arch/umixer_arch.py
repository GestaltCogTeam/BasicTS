"""UMixer forecasting architecture."""

import torch
import torch.fft
import torch.nn as nn

from basicts.modules.embed.calendar_embed import \
    ConvolutionalPatchEmbedding as PatchEmbedding
from basicts.modules.norm import RevIN

from ..config.umixer_config import UMixerConfig


def S_Correction(x, x_pre):
    x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
    x_pre_fft = torch.fft.rfft(x_pre, dim=1, norm="ortho")
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    x_ifft = torch.fft.irfft(x_fft, dim=1)
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=1)
    x_ifft = torch.clamp(x_ifft, min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft, min=0)
    alpha = torch.sum(x_ifft * x_pre_ifft, dim=1, keepdim=True) / (
        torch.sum(x_pre_ifft * x_pre_ifft, dim=1, keepdim=True) + 0.001
    )
    return torch.sqrt(alpha)


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class channelMix_CI_pat(nn.Module):
    def __init__(self, config: UMixerConfig):
        super(channelMix_CI_pat, self).__init__()
        self.pred_len = config.output_len
        self.seq_len = config.input_len
        self.stride = config.stride
        self.patch_len = config.patch_len
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.Pnum = int(
            (self.pred_len + self.seq_len - self.patch_len) / self.stride + 2
        )
        self.conv1 = nn.ModuleList(
            (nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        )
        self.conv2 = nn.ModuleList(
            (nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        )
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.d_model

    def forward(self, x):
        o = torch.zeros_like(x)
        for i in range(self.channels):
            o[:, :, i] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, i]))))
        res = o + x
        res = self.norm(res)
        return res


class tempolMix_CI_pat(nn.Module):
    def __init__(self, config: UMixerConfig):
        super(tempolMix_CI_pat, self).__init__()
        self.pred_len = config.output_len
        self.seq_len = config.input_len
        self.stride = config.stride
        self.patch_len = config.patch_len
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.Pnum = int(
            (self.pred_len + self.seq_len - self.patch_len) / self.stride + 2
        )
        self.conv1 = nn.ModuleList(
            (nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        )
        self.conv2 = nn.ModuleList(
            (nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        )
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.Pnum

    def forward(self, x):
        o = torch.zeros_like(x)
        for i in range(self.channels):
            o[:, i, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, i, :]))))
        res = o + x
        res = self.norm(res)
        return res


class UMixer(nn.Module):
    """
    Paper: U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting
    Link: https://arxiv.org/abs/2401.02236
    Ref Official Code: https://github.com/XiangMa-Shaun/U-Mixer
    Venue: AAAI 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: UMixerConfig):
        super(UMixer, self).__init__()
        self.pred_len = config.output_len
        self.seq_len = config.input_len
        self.stride = config.stride
        self.patch_len = config.patch_len
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.Pnum = int(
            (self.pred_len + self.seq_len - self.patch_len) / self.stride + 2
        )
        self.layer = config.num_layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.e_layers = config.num_layers
        self.d_layers = 1
        self.enc_in = config.num_features
        self.dropout = config.dropout
        self.c_out = config.num_features
        self.mlp_tempmix_md = nn.ModuleList(
            [tempolMix_CI_pat(config) for _ in range(self.e_layers)]
        )
        self.mlp_chanmix_md = nn.ModuleList(
            [channelMix_CI_pat(config) for _ in range(self.e_layers)]
        )
        self.mlp_tempmix_mu = nn.ModuleList(
            [tempolMix_CI_pat(config) for _ in range(self.e_layers)]
        )
        self.mlp_chanmix_mu = nn.ModuleList(
            [channelMix_CI_pat(config) for _ in range(self.e_layers)]
        )
        self.mlp_trend_ci = nn.ModuleList(
            (nn.Linear(self.pred_len, self.d_model) for _ in range(self.c_out))
        )
        self.mlp_trend2_ci = nn.ModuleList(
            (nn.Linear(self.d_model, self.pred_len) for _ in range(self.c_out))
        )
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(self.enc_in, eps=1e-5)
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.dropout
        )
        self.head = Flatten_Head(
            self.enc_in,
            self.d_model * self.Pnum,
            self.pred_len,
            head_dropout=self.dropout,
        )
        self.comb = nn.Linear(self.e_layers, 1)
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def forecast(self, x_input, reference_inputs=None):
        # The correction branch uses unnormalized history in the source model.
        # An external normalizer supplies that reference separately.
        x_ori = (x_input if reference_inputs is None else reference_inputs).contiguous()
        if self.use_revin:
            x_input = self.revin(x_input, "norm")
        x_input = self.predict_linear(x_input.permute(0, 2, 1))
        (x_input, n_vars) = self.patch_embedding(x_input)
        (x_old, _) = self.patch_embedding(x_ori.permute(0, 2, 1))
        x_all = x_input.new_zeros(
            x_input.shape[0], x_input.shape[1], x_input.shape[2], self.layer
        )
        for i in range(self.layer):
            x_ud = self.mlp_tempmix_md[i](x_input)
            x_ud = self.mlp_chanmix_md[i](x_ud)
            for i in range(i, -1, -1):
                x_ud = self.mlp_tempmix_mu[i](x_ud)
                x_ud = self.mlp_chanmix_mu[i](x_ud)
            x_all[:, :, :, i] = x_ud
        x_input = self.comb(x_all).squeeze(-1)
        x_input = (
            S_Correction(
                self.layer_norm(x_old), self.layer_norm(x_input[:, : x_old.shape[1], :])
            )
            * x_input
        )
        x_input = torch.reshape(
            x_input, (-1, n_vars, x_input.shape[-2], x_input.shape[-1])
        )
        x_input = x_input.permute(0, 1, 3, 2)
        x_input = self.head(x_input)
        x_input = x_input.permute(0, 2, 1)
        if self.use_revin:
            x_input = self.revin(x_input, "denorm")
        x = x_input[:, -self.pred_len :, :]
        return x

    def forward(self, inputs, reference_inputs=None) -> torch.Tensor:
        """

        Args:
            inputs (Tensor): Input data with shape [B, input_len, num_features].

        Returns:
            torch.Tensor: outputs with shape [B, output_len, num_features]
        """
        x_input = inputs
        out = self.forecast(x_input, reference_inputs=reference_inputs)
        return out
