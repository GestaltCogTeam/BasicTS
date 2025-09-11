import torch
import torch.nn as nn

from argparse import Namespace

class RecurrentCycle(torch.nn.Module):
    """
    From the author:
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    """
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index.long()]


class CycleNet(nn.Module):
    """
        Paper: CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns
        Link: https://arxiv.org/pdf/2409.18479
        Official Code: https://github.com/ACAT-SCUT/CycleNet
        Venue:  NIPS 2024
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(CycleNet, self).__init__()
        configs = Namespace(**model_args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycle_pattern = configs.cycle_pattern

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        x = history_data[..., 0]

        if self.cycle_pattern == 'daily':
            cycle_index = history_data[..., 1] * self.cycle_len # [B]
            cycle_index = cycle_index[:, -1, 0] # from CycleNet data_loader.py: "cycle_index = torch.tensor(self.cycle_index[s_end])""
        elif self.cycle_pattern == 'daily&weekly':
            cycle_index = history_data[..., 1] * self.cycle_len * 7 + history_data[..., 2] * 7
            cycle_index = cycle_index[:, -1, 0]
        else:
            raise Exception("please specify cycle pattern, daily OR weekly OR others")

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y.unsqueeze(-1)

