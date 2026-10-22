"""OLinear channel learner, extracted unchanged from the author implementation."""

import math
import random
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
        )

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


class LinearEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff=None,
        CovMat=None,
        dropout=0.1,
        activation="relu",
        token_num=None,
        **kwargs,
    ):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = (
            torch.eye(self.token_num) * 1.0
            + torch.randn(self.token_num, self.token_num) * 1.0
        )
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        # self.bias = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
        values = self.v_proj(x)

        if self.CovMat is not None:
            A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            A = F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)

        new_x = A @ values  # + self.bias

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class Encoder_ori(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers=None,
        norm_layer=None,
        one_output=False,
        CKA_flag=False,
    ):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag
        if self.CKA_flag:
            print("CKA is enabled...")

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, nvars, D]
        attns = []
        X0 = None  # to make Pycharm happy
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0:
                    X0 = x

                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    CudaCKA1 = CudaCKA(device=x.device)
                    cka_value = CudaCKA1.linear_CKA(
                        X0.flatten(0, 1)[:1000], x.flatten(0, 1)[:1000]
                    )
                    print(f"CKA: \t{cka_value:.3f}")

        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns
