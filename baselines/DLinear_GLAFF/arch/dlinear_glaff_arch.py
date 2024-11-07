import torch
import torch.nn as nn
from .glaff import Plugin

import pdb

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear_GLAFF(nn.Module):
    """
        Paper: Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective
        Link: https://arxiv.org/pdf/2409.18696
        Official Code: https://github.com/ForestsKing/GLAFF
        Venue: NIPS 2024
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(DLinear_GLAFF, self).__init__()
        self.hist_len = model_args["hist_len"]
        self.pred_len = model_args["pred_len"]
        self.glaff = model_args['glaff']
        self.time_of_day_size = model_args["time_of_day_size"]

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = model_args["individual"]
        self.channels = model_args["enc_in"]

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.hist_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.hist_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.hist_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.hist_len, self.pred_len)

        if self.glaff:
            self.plugin = Plugin(**model_args)

    def get_time_feature(self, time_features):
        '''
        处理时间特征
        如果能直接使用dataframe的index会更方便，这里是折中的方案
        除非有秒级的数据，不然second应该都是一样的

        '''
        month = (time_features[..., 2][..., 0] * 31 / 12 - 0.5).unsqueeze(-1)
        day = (time_features[..., 2][..., 0] - 0.5).unsqueeze(-1)
        weekday = (time_features[..., 1][..., 0] * 7 / 6 - 0.5).unsqueeze(-1)
        hour = (time_features[..., 0][..., 0] * 24 / 23 - 0.5).unsqueeze(-1)
        minute = (time_features[..., 0][..., 0] - 0.5).unsqueeze(-1)
        second = (torch.zeros_like(time_features[..., 0][..., 0]) - 0.5).unsqueeze(-1)
        time = torch.cat([month, day, weekday, hour, minute, second], dim=-1)
        return time

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """ 
        x = history_data[..., 0]     # B, L, N
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(
                1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(
                1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction = seasonal_output + trend_output

        if self.glaff:
            x_time = self.get_time_feature(history_data[..., 1:])
            y_time = self.get_time_feature(future_data[..., 1:])
            label_len = self.hist_len // 2
            map1 = prediction.clone().transpose(1, 2)
            x_enc_copy = x.clone()
            x_mark_enc_copy = x_time
            x_mark_dec_copy = torch.concat([x_time[:, -label_len:, :], y_time], dim=1)
            prediction, reco, map1 = self.plugin(x_enc_copy, x_mark_enc_copy, map1, x_mark_dec_copy[:, -self.pred_len:, :])
            return prediction.unsqueeze(-1)  # [B, L, N, 1]

        else:
            return prediction.unsqueeze(-1)  # [B, L, N, 1]
