import torch
from torch import nn


class RMSNorm(nn.Module):

    """
    Root Mean Square Layer Normalization
    """

    def __init__(self, hidden_size: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states
