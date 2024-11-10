import torch
import torch.nn as nn

from .glaff import Plugin


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
        NOTE: You may find that the results produced by BasicTS differ from those in the original paper, mainly due to the following reasons:
                1. There is an issue of data leakage in the data processing part of the original code.
                    It uses the entire dataset, instead of only the training data, for normalization, 
                        which can lead to some differences in results, especially on the Traffic and ETTm2 datasets.
                    See: https://github.com/ForestsKing/GLAFF/blob/c91679403b879c892f21c22eb0f53f314711d6f0/data/dataset.py#L18
                2. The experimental setup in the original paper differs from common practices.
                    For example, ETTm2 typically uses only the first 20 months of data, whereas the original code uses all data.
                    Additionally, the original code splits the training, testing, and validation sets in a 6:2:2 ratio for all datasets,
                        while the Traffic dataset is usually split in a 7:1:2 ratio.
                It is important to note that despite these slight differences, GLAFF remains effective.
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
                self.Linear_Seasonal.append(nn.Linear(self.hist_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.hist_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))

        else:
            self.Linear_Seasonal = nn.Linear(self.hist_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.hist_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter((1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.hist_len) * torch.ones([self.pred_len, self.hist_len]))

        if self.glaff:
            self.plugin = Plugin(**model_args)

    def _day_of_year_to_month_day_tensor(self, day_of_year_tensor, is_leap_year=False):
        # 定义每月的天数
        days_in_month = torch.tensor(
            [31, 29 if is_leap_year else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        )
        cumulative_days = torch.cumsum(days_in_month, dim=0)  # 获取每月的累计天数
        
        # 初始化输出张量
        month_tensor = torch.zeros_like(day_of_year_tensor)
        day_tensor = torch.zeros_like(day_of_year_tensor)
        
        # 遍历每月的累计天数，确定 day_of_year 属于哪个月份
        for month in range(12):
            if month == 0:
                mask = (day_of_year_tensor <= cumulative_days[month])
            else:
                mask = (day_of_year_tensor > cumulative_days[month - 1]) & (day_of_year_tensor <= cumulative_days[month])
            
            # 将符合条件的元素赋值为对应的月份和日期
            month_tensor[mask] = month + 1
            day_tensor[mask] = day_of_year_tensor[mask] - (cumulative_days[month - 1] if month > 0 else 0)
        
        return month_tensor, day_tensor

    def timeslot_to_time_tensor(self, time_of_day_tensor, time_of_day_size):
        # 一天的总秒数
        total_seconds_per_day = 86400
        # 每个时间片的秒数
        seconds_per_timeslot = total_seconds_per_day // time_of_day_size

        # 当前时间片对应的总秒数（张量运算）
        total_seconds_tensor = time_of_day_tensor * seconds_per_timeslot

        # 计算小时、分钟和秒
        hours_tensor = total_seconds_tensor // 3600
        minutes_tensor = (total_seconds_tensor % 3600) // 60
        seconds_tensor = total_seconds_tensor % 60

        return hours_tensor, minutes_tensor, seconds_tensor

    def get_time_feature(self, time_features):
        '''
        处理时间特征
        如果能直接使用dataframe的index会更方便，这里是折中的方案
        除非有秒级的数据，不然second应该都是一样的
        '''
        time_features = time_features[:, :, 0, :]

        # time of day to seconds, minutes, hours
        time_of_day = time_features[..., 0] * self.time_of_day_size
        hours, minuts, seconds = self.timeslot_to_time_tensor(time_of_day, self.time_of_day_size)
        
        # day of year to month of year, day of month, day of week
        day_of_year = time_features[..., 3] * 366
        month, day = self._day_of_year_to_month_day_tensor(day_of_year)
        weekday = time_features[..., 1] * 7

        # generate the timestamp features required by the model
        
        month = month / 12 - 0.5
        day = day / 31 - 0.5
        weekday = weekday / 6 - 0.5
        hours = hours / 23 - 0.5
        minuts = minuts / 59 - 0.5
        seconds = seconds / 59 - 0.5

        time = torch.stack([month, day, weekday, hours, minuts, seconds], dim=-1).clone()
        return time

    def dlinear_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()
        x = x_enc

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        dec_out = seasonal_output + trend_output
        dec_out = dec_out.permute(0, 2, 1)
        pred = dec_out[:, -self.pred_len:, :]

        if self.glaff:
            map1 = pred.clone()
            pred, reco, map2 = self.plugin(x_enc_copy, x_mark_enc_copy, map1, x_mark_dec_copy[:, -self.pred_len:, :])
            return pred, reco, map1, map2
        else:
            return pred, None, None, None

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """ 
        x_time = self.get_time_feature(history_data[..., 1:]).clone()
        y_time = self.get_time_feature(future_data[..., 1:]).clone()
        x_data = history_data[..., 0].clone()
        y_data = future_data[..., 0].clone()
        
        label_len = self.hist_len // 2
        x_enc = x_data
        x_mark_enc = x_time
        x_dec = torch.cat([x_data[:, -label_len:, :], torch.zeros_like(y_data)], dim=1)
        x_mark_dec = torch.cat([x_time[:, -label_len:, :], y_time], dim=1)
        
        pred, reco, map1, map2 = self.dlinear_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.glaff:
            # construct for plugin loss
            plugin_prediction = torch.cat([pred, reco, map1, map2], dim=1)
            plugin_target = torch.cat([y_data, x_data, y_data, y_data], dim=1)
            
            return {
                "prediction": pred.unsqueeze(-1),  # [B, L, N, 1]
                "plugin_prediction": plugin_prediction,
                "plugin_target": plugin_target
            }
        else:
            return {
                "prediction": pred.unsqueeze(-1),  # [B, L, N, 1]
                "plugin_prediction": pred.unsqueeze(-1),
                "plugin_target": y_data.unsqueeze(-1)
            }