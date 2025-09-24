from typing import Optional

import torch
from torch import nn


class PatchTSTBatchNorm(nn.Module):
    """
    BatchNorm layer for PatchTST.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x.transpose(1, 2))
        return x.transpose(1, 2)


class PatchTSTHead(nn.Module):
    """
    Head layer for PatchTST.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            individual: bool = False,
            num_features: Optional[int] = None,
            dropout: float = 0.0):
        super().__init__()
        self.individual = individual
        self.num_features = num_features
        # Warning: classification task should not use individual head.
        if self.individual:
            if self.num_features is None:
                raise ValueError("num_features is required when individual is True")
            self.linears = nn.ModuleList(
                [nn.Linear(input_size, output_size) for _ in range(num_features)])
            self.dropouts = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(num_features)])
        else:
            self.linear = nn.Linear(input_size, output_size)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual: # individual head
            if self.num_features != x.shape[1]:
                raise ValueError(
                    f"num_features ({self.num_features}) does not match input shape ({x.shape[1]})." \
                    " Warning: classification task should not use individual head.")
            x_out = []
            for i in range(self.num_features):
                out = self.dropouts[i](self.linears[i](x[:, i, :]))
                x_out.append(out)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.dropout(self.linear(x))
        return x
