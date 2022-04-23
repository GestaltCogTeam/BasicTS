import torch
import torch.nn as nn

from basicts.archs.BasicMTS_arch.MLP import MLP_res

class DecoderNN(nn.Module):
    def __init__(self, hidden_dim, out_dim) -> None:
        super().__init__()
        self.mlp = MLP_res(hidden_dim, hidden_dim)
        self.regression_layer = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1,1), bias=True)

    def forward(self, hidden:torch.Tensor) -> torch.Tensor:
        """feed forward of MLP.

        Args:
            hidden (torch.Tensor): hidden representation with shape [B, D, N, 1]

        Returns:
            torch.Tensor: latent representation [B, D, N, 1]
        """
        B, D, N, _ = hidden.shape
        # regression
        hidden = self.mlp(hidden)
        prediction = self.regression_layer(hidden)
        return prediction
