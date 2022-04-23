import torch
import torch.nn as nn

class MLP_res(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1,1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,1), bias=True)
        self.act = nn.SiLU()

    def forward(self, input_data:torch.Tensor) -> torch.Tensor:
        """feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        B, D, N, _ = input_data.shape
        hidden  = self.fc2(self.act(self.fc1(input_data)))      # MLP
        hidden  = hidden + input_data                           # residual
        return hidden
