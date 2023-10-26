import torch
import torch.nn as nn

class NLinear(nn.Module):
    """
    Paper: Are Transformers Effective for Time Series Forecasting?
    Link: https://arxiv.org/abs/2205.13504
    Official Code: https://github.com/cure-lab/DLinear
    """

    def __init__(self, **model_args):
        super(NLinear, self).__init__()
        self.seq_len = model_args["seq_len"]
        self.pred_len = model_args["pred_len"]
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of NLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        assert history_data.shape[-1] == 1      # only use the target feature
        x = history_data[..., 0]     # B, L, N
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        prediction = x + seq_last
        return prediction.unsqueeze(-1)
