"""OLinear native forecasting model with registered orthogonal bases."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basicts.modules.norm import RevIN
from .olinear_layers import Encoder_ori, LinearEncoder
from ..config.olinear_config import OLinearConfig


def _dct_matrix(size: int) -> np.ndarray:
    mat = np.empty((size, size), dtype=np.float32)
    scale0 = np.sqrt(1.0 / size)
    scale = np.sqrt(2.0 / size)
    n = np.arange(size, dtype=np.float32)
    for k in range(size):
        mat[k] = (scale0 if k == 0 else scale) * np.cos(np.pi * (n + 0.5) * k / size)
    return mat


class OLinear(nn.Module):
    """
    Paper: OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain
    Link: https://papers.neurips.cc/paper_files/paper/2025/hash/571c7e164fb1ffbcf2f84a63784451ec-Abstract-Conference.html
    Ref Official Code: https://github.com/jackyue1994/OLinear
    Venue: NeurIPS 2025
    Task: Time Series Forecasting

    Note:
        Supply train-only correlation bases through input_basis_path/output_basis_path.
        When omitted, the model uses the fixed DCT basis variant.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: OLinearConfig):
        super(OLinear, self).__init__()

        def basis(length, path):
            # Data-dependent bases must be computed from training data only.
            # Explicit paths must exist; never silently substitute another basis.
            matrix = np.load(path) if path is not None else _dct_matrix(length)
            if config.channel_independent_basis and matrix.ndim == 2:
                matrix = np.broadcast_to(
                    matrix, (config.num_features, length, length)
                ).copy()
            expected = (
                (config.num_features, length, length)
                if config.channel_independent_basis
                else (length, length)
            )
            if matrix.shape != expected or not np.isfinite(matrix).all():
                raise ValueError(
                    f"Expected finite basis with shape {expected}, got {matrix.shape}"
                )
            return matrix

        q_matrix = basis(config.input_len, config.input_basis_path)
        q_out_matrix = basis(config.output_len, config.output_basis_path)
        self.pred_len = config.output_len
        self.enc_in = config.num_features
        self.seq_len = config.input_len
        self.hidden_size = self.d_model = config.hidden_size
        self.d_ff = config.intermediate_size
        self.Q_chan_indep = config.channel_independent_basis
        self.register_buffer("Q_mat", torch.from_numpy(q_matrix).to(torch.float32))
        assert self.Q_mat.ndim == 3 if self.Q_chan_indep else self.Q_mat.ndim == 2
        assert (
            self.Q_mat.shape[0] == self.enc_in
            if self.Q_chan_indep
            else self.Q_mat.shape[0] == self.seq_len
        )
        self.register_buffer(
            "Q_out_mat", torch.from_numpy(q_out_matrix).to(torch.float32)
        )
        assert (
            self.Q_out_mat.ndim == 3 if self.Q_chan_indep else self.Q_out_mat.ndim == 2
        )
        assert (
            self.Q_out_mat.shape[0] == self.enc_in
            if self.Q_chan_indep
            else self.Q_out_mat.shape[0] == self.pred_len
        )
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.embed_size = config.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len),
        )
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin_layer = RevIN(self.enc_in, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    CovMat=None,
                    dropout=config.dropout,
                    activation=config.hidden_act,
                    token_num=self.enc_in,
                )
                for _ in range(config.num_layers)
            ],
            norm_layer=nn.LayerNorm(config.hidden_size),
            one_output=True,
            CKA_flag=config.cka,
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size),
        )
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        return x * embeddings

    def Fre_Trans(self, x):
        (B, N, T, D) = x.shape
        assert T == self.seq_len
        x = x.transpose(-1, -2)
        if self.Q_chan_indep:
            x_trans = torch.einsum("bndt,ntv->bndv", x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = (
                torch.einsum("bndt,tv->bndv", x, self.Q_mat.transpose(-1, -2))
                + self.delta1
            )
        assert x_trans.shape[-1] == self.seq_len
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)
        if self.Q_chan_indep:
            x = torch.einsum("bndt,ntv->bndv", x_trans, self.Q_out_mat)
        else:
            x = torch.einsum("bndt,tv->bndv", x_trans, self.Q_out_mat) + self.delta2
        x = x.transpose(-1, -2)
        return x

    def forward(self, inputs):
        x = inputs
        (B, T, N) = x.shape
        if self.use_revin:
            x = self.revin_layer(x, mode="norm")
        x_ori = x
        x = self.tokenEmb(x_ori, self.embeddings)
        x = self.Fre_Trans(x)
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        if self.use_revin:
            out = self.revin_layer(out, mode="denorm")
        return out
