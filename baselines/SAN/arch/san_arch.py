import torch
import torch.nn as nn
from .dlinear import DLinear
from argparse import Namespace
import pdb 

class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        configs = Namespace(**configs)
        self.seq_len = configs.seq_len // configs.period_len
        self.pred_len = int(configs.pred_len / configs.period_len)
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)


class SAN(nn.Module):
    """
        Paper: Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective
        Link: 
        Official Code: 
        Venue: NIPS 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(SAN, self).__init__()
        self.seq_len = model_args["seq_len"]
        self.pred_len = model_args["pred_len"]
        self.period_len = model_args["period_len"]
        self.station_pretrain_epoch = model_args["station_pretrain_epoch"]

        self.channels = model_args["enc_in"]
        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self.weight = nn.Parameter(torch.ones(2, self.channels))

        self.backbone = DLinear(**model_args)
        self.model = MLP(model_args, mode='mean')
        self.model_std = MLP(model_args, mode='std')
        
    def normalize(self, x):
        bs, length, dim = x.shape # (B, L, N)
        x = x.reshape(bs, -1, self.period_len, dim)
        mean = torch.mean(x, dim=-2, keepdim=True)
        std = torch.std(x, dim=-2, keepdim=True)
        norm_x = (x - mean) / (std + self.epsilon)
        x = x.reshape(bs, length, dim)
        mean_all = torch.mean(x, dim=1, keepdim=True)

        outputs_mean = self.model(mean.squeeze(2) - mean_all, x - mean_all) * self.weight[0] + mean_all * self.weight[1]
        outputs_std = self.model_std(std.squeeze(2), x)
        outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
        return norm_x.reshape(bs, length, dim), outputs[:, -self.pred_len_new:, :]

    def de_normalize(self, y, station_pred):
        bs, length, dim = y.shape
        y = y.reshape(bs, -1, self.period_len, dim)
        mean = station_pred[:, :, :self.channels].unsqueeze(2)
        std = station_pred[:, :, self.channels:].unsqueeze(2)
        output = y * (std + self.epsilon) + mean
        return output.reshape(bs, length, dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        assert history_data.shape[-1] == 1      # only use the target feature
        target = future_data[..., 0]
        x = history_data[..., 0]     # B, L, N
        x, statistics_pred = self.normalize(x)
        y = self.backbone(x)
        y = self.de_normalize(y, statistics_pred)

        return {"prediction": y.unsqueeze(-1), 
                "statistics_pred":statistics_pred, 
                "period_len":self.period_len, "epoch": epoch,
                "station_pretrain_epoch": self.station_pretrain_epoch,
                "train": train}

