import torch
import torch.nn as nn

from .Invertible import RevIN


class RLinear(nn.Module):
    """
    Paper: Revisiting Long-Term Time Series Forecasting: An Investigation on Linear Mapping
    Link: https://arxiv.org/abs/2305.10721
    Official Code: https://github.com/plumprc/RTSF
    """
    def __init__(self, seq_len,pred_len,channel,drop,rev,individual):
        super(RLinear, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(channel)
        ]) if individual else nn.Linear(seq_len, pred_len)

        self.dropout = nn.Dropout(drop)
        self.rev = RevIN(channel) if rev else None
        self.individual = individual

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of RLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        x = history_data[:,:,:,0]
        y = future_data[:,:,:,0]
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        return pred.unsqueeze(-1)