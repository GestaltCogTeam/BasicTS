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
