import torch
from torch import nn


class STNorm(nn.Module):
    """
    Paper: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
    Link: https://dl.acm.org/doi/10.1145/3447548.3467330
    Ref Official Code: https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py
    Venue: SIGKDD 2021
    Task: Spatial-Temporal Forecasting
    """
    def __init__(
            self,
            hidden_size: int,
            enable_snorm: bool = True,
            enable_tnorm: bool = True,
            num_features: int = None,
            momentum: float = 0.1,
            eps: float = 1e-6):
        super().__init__()
        self.enable_snorm = enable_snorm
        self.enable_tnorm = enable_tnorm
        self.eps = eps

        if self.enable_snorm:
            self.beta_s = nn.Parameter(torch.zeros(1, hidden_size, 1, 1))
            self.gamma_s = nn.Parameter(torch.ones(1, hidden_size, 1, 1))
        if self.enable_tnorm:
            self.beta_t = nn.Parameter(torch.zeros(1, hidden_size, num_features, 1))
            self.gamma_t = nn.Parameter(torch.ones(1, hidden_size, num_features, 1))
            self.register_buffer(
                "running_mean", torch.zeros(1, hidden_size, num_features, 1))
            self.register_buffer(
                "running_var", torch.ones(1, hidden_size, num_features, 1))
            self.momentum = momentum

    def snorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalization over the spatial dimension.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, hidden_size, num_features, num_timestamps + 1].

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(dim=2, keepdims=True)
        std = x.std(dim=2, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        x_norm = x_norm * self.gamma_s + self.beta_s
        return x_norm

    def tnorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal normalization.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, hidden_size, num_features].

        Returns:
            torch.Tensor: Normalized tensor with shape [batch_size, hidden_size, num_features].
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        mean = x.mean((0, 3), keepdims=True)
        std = x.std((0, 3), keepdims=True, unbiased=False)
        if self.training:
            n = x.shape[3] * x.shape[0]
            with torch.no_grad():
                self.running_mean = self.momentum * mean + \
                    (1 - self.momentum) * self.running_mean
                self.running_std = self.momentum * std * n / \
                    (n - 1) + (1 - self.momentum) * self.running_std
        else:
            mean = self.running_mean
            std = self.running_std
        x_norm = (x - mean) / (std + self.eps)
        x_norm = x_norm * self.gamma_t + self.beta_t
        return x_norm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of STNorm.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            or [batch_size, hidden_size, num_features, num_timestamps + 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: normalized tensors with spatial and temporal normalization.
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if self.enable_snorm:
            x_snorm = self.snorm(x)
            if x.ndim == 3:
                x_snorm = x_snorm.squeeze(-1)
        else:
            x_snorm = None
        if self.enable_tnorm:
            x_tnorm = self.tnorm(x)
            if x.ndim == 3:
                x_tnorm = x_tnorm.squeeze(-1)
        else:
            x_tnorm = None
        return x_snorm, x_tnorm
