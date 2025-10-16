import torch
from torch import nn


class CenteredLayerNorm(nn.Module):
    """
    Centered LayerNorm with zero mean, special designed for the seasonal part of time series.
    Original implementation is from Autoformer.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layernorm(x)
        bias = torch.mean(x_norm, dim=1, keepdim=True)
        return x_norm - bias
