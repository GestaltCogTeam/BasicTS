import torch
import torch.nn as nn
from .Embed import PositionalEmbedding

class SparseTSF(nn.Module):
    """
    Paper: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters
    Official Code: https://github.com/lss-1138/SparseTSF
    Link: https://arxiv.org/abs/2405.00946
    Venue: ICML 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(SparseTSF, self).__init__()

        # get parameters
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.enc_in = model_args['enc_in']
        self.period_len = model_args['period_len']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        x = history_data[:, :, :, 0]
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y.unsqueeze(-1)
