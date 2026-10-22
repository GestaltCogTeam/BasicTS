"""Fredformer forecasting architecture."""

from ..config.fredformer_config import FredformerConfig
from typing import Callable, Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .Fredformer_backbone import Flatten_Head
from basicts.modules.norm import RevIN
from .cross_Transformer_nys import Trans_C as Trans_C_nys
from .cross_Transformer import Trans_C


class Fredformer(nn.Module):
    """
    Paper: Fredformer: Frequency Debiased Transformer for Time Series Forecasting
    Link: https://arxiv.org/abs/2406.09009
    Ref Official Code: https://github.com/chenzRG/Fredformer
    Venue: KDD 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: FredformerConfig):
        super(Fredformer, self).__init__()
        ablation = config.ablation
        mlp_drop = config.mlp_dropout
        use_nys = int(config.use_nystrom)
        output = config.output
        mlp_hidden = config.mlp_hidden_size
        cf_dim = config.cf_dim
        cf_depth = config.cf_depth
        cf_heads = config.cf_heads
        cf_mlp = config.cf_mlp
        cf_head_dim = config.cf_head_dim
        cf_drop = config.cf_dropout
        c_in = config.num_features
        context_window = config.input_len
        target_window = config.output_len
        patch_len = config.patch_len
        stride = config.stride
        d_model = config.hidden_size
        head_dropout = config.head_dropout
        padding_patch = config.padding_patch
        individual = config.individual
        revin = config.use_revin
        affine = config.affine
        subtract_last = config.subtract_last
        self.use_nys = use_nys
        self.ablation = ablation
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                c_in, affine=affine, subtract_last=subtract_last, eps=1e-5
            )
        self.output = output
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len) / stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        self.re_attn = True
        if self.use_nys == 0:
            self.fre_transformer = Trans_C(
                dim=cf_dim,
                depth=cf_depth,
                heads=cf_heads,
                mlp_dim=cf_mlp,
                dim_head=cf_head_dim,
                dropout=cf_drop,
                patch_dim=patch_len * 2,
                horizon=self.horizon * 2,
                d_model=d_model * 2,
            )
        else:
            self.fre_transformer = Trans_C_nys(
                dim=cf_dim,
                depth=cf_depth,
                heads=cf_heads,
                mlp_dim=cf_mlp,
                dim_head=cf_head_dim,
                dropout=cf_drop,
                patch_dim=patch_len * 2,
                horizon=self.horizon * 2,
                d_model=d_model * 2,
            )
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            target_window,
            head_dropout=head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            target_window,
            head_dropout=head_dropout,
        )
        self.ircom = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow * 2 - 2, self.targetwindow)
        self.final = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.get_r = nn.Linear(d_model * 2, d_model * 2)
        self.get_i = nn.Linear(d_model * 2, d_model * 2)
        self.output1 = nn.Linear(target_window, target_window)
        self.input = nn.Linear(c_in, patch_len * 2)
        self.outpt = nn.Linear(d_model * 2, c_in)
        self.abfinal = nn.Linear(patch_len * patch_num, target_window)
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def forward(self, inputs) -> torch.Tensor:
        """

        Args:
            inputs (Tensor): Input data with shape [B, input_len, num_features].

        Returns:
            torch.Tensor: outputs with shape [B, output_len, num_features]
        """
        z = inputs
        if self.revin:
            z = self.revin_layer(z, "norm")
        z = z.transpose(1, 2)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)
        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)
        z = torch.fft.ifft(torch.complex(z1.float(), z2.float()))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))
        z = z.transpose(1, 2)
        if self.revin:
            z = self.revin_layer(z, "denorm")
        return z
