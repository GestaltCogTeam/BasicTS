# pylint: disable=not-callable
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal, special
from torch import nn

from basicts.modules.norm import RevIN

from ..config.film_config import FiLMConfig


class HippoProj(nn.Module):
    """
    Hippo projection layer.
    """
    def __init__(
            self,
            order_hippo: int,
            discretization_timestep: float = 1.0,
            discretization_method: str = "bilinear"):
        """
        order_hippo: the order of the HiPPO projection
        discretization step size: It should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.order_hippo = order_hippo
        Q = np.arange(order_hippo, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
        B = (-1.) ** Q[:, None] * R
        C = np.ones((1, self.order_hippo))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete(
            (A, B, C, D), dt = discretization_timestep, method = discretization_method)
        B = B.squeeze(-1)

        self.register_buffer("A", torch.Tensor(A))
        self.register_buffer("B", torch.Tensor(B))
        vals = np.arange(0.0, 1.0, discretization_timestep)
        self.register_buffer(
            "eval_matrix",
            torch.Tensor(
                special.eval_legendre(np.arange(self.order_hippo)[:, None], 1 - 2 * vals).T)
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.order_hippo])).to(inputs.device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            new = new.to(inputs.device)
            c = F.linear(c, self.A.to(inputs.device)) + new
            cs.append(c)

        return torch.stack(cs, dim=0)


class SpectralConv1d(nn.Module):
    """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    """
    def __init__(self, input_size: int, output_size: int, input_len: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.modes = min(32, input_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = 1 / (self.input_size * self.output_size)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(self.input_size, self.output_size, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(self.input_size, self.output_size, len(self.index), dtype=torch.float))

    def compl_mul1d(
            self,
            order: str,
            x: torch.Tensor,
            weights_real: torch.Tensor,
            weights_imag: torch.Tensor
            ) -> torch.Tensor:
        return torch.complex(
            torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
            torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, _, _ = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            B, H, self.output_size, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        x_ft = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d(
            "bjix,iox->bjox", x_ft, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FiLM(nn.Module):
    """
    Paper: FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting
    Official Code: https://github.com/tianzhou2011/FiLM
    Link: https://arxiv.org/abs/2205.08897
    Venue: NeurIPS 2022
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: FiLMConfig):
        super().__init__()

        self.input_len = config.input_len
        self.output_len = config.output_len
        self.use_revin = config.use_revin
        # b, s, f means b, f
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=True)

        self.multiscale = config.multiscale
        self.hidden_size = config.hidden_size
        self.legts = nn.ModuleList([
            HippoProj(
                order_hippo = hidden_size,
                discretization_timestep = 1. / self.output_len / i
                ) for hidden_size in self.hidden_size for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([
            SpectralConv1d(
                input_size = hidden_size,
                output_size = hidden_size,
                input_len = min(self.output_len, self.input_len)
                ) for hidden_size in self.hidden_size for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.hidden_size), 1)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feed forward of FiLM.

        Args:
            inputs (torch.Tensor): inputs data with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: prediction with shape [batch_size, output_len, num_features]
        """

        # Normalization
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # Backbone
        prediction = []
        for i in range(0, len(self.multiscale) * len(self.hidden_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.output_len
            x_in = inputs[:, -x_in_len:]
            legt = self.legts[i]
            x_in = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])
            x_out = self.spec_conv_1[i](x_in)
            if self.input_len >= self.output_len:
                x_out = x_out.transpose(2, 3)[:, :, self.output_len - 1, :]
            else:
                x_out = x_out.transpose(2, 3)[:, :, -1, :]
            x_out = x_out @ legt.eval_matrix[-self.output_len:, :].T
            prediction.append(x_out)

        prediction = torch.stack(prediction, dim=-1)
        prediction = self.mlp(prediction).squeeze(-1).permute(0, 2, 1)

        # De-Normalization
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        return prediction
