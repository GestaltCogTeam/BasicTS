import torch
import torch.nn as nn
from argparse import Namespace
from .s4d import S4D

import pdb 

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4(nn.Module):
    """
        Paper: EFFICIENTLY MODELING LONG SEQUENCES WITH STRUCTURED STATE SPACES
        Link: https://openreview.net/pdf?id=uYLFoz1vlAC
        Official Code: https://github.com/state-spaces/s4
        Venue: ICLR 2022
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(S4, self).__init__()

        config = Namespace(**model_args)
        self.prenorm = config.prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(config.seq_len, config.d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(config.n_layers):
            self.s4_layers.append(
                S4D(config.d_model, dropout=config.dropout, transposed=True, lr=min(0.001, config.lr))
            )
            self.norms.append(nn.LayerNorm(config.d_model))
            self.dropouts.append(dropout_fn(config.dropout))

        # Linear decoder
        self.decoder = nn.Linear(config.d_model, config.d_output)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of S4.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]
            S4 Input x is shape (B, L, d_input)
        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        """

        """
        x = history_data[..., 0]
        x = x.transpose(-1, -2)  # (B, L, d_input) -> (B, d_input, L)
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_input, d_model)

        x = x.transpose(-1, -2)  # (B, d_input, d_model) -> (B, d_model, d_input)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)


        x = x.transpose(-1, -2)

        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x.permute(0, 2, 1).unsqueeze(-1)  # [B, L, N, 1]
