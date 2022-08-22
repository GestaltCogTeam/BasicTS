"""
    Statistical models including: MA (Moveing Average) AR (Auto Regression), VAR (Vector Auto Regression), and ARIMA (Autoregressive Integrated Moving Average (ARIMA).
                                    All the random noise term is omitted.
    Ref Code: https://github.com/doowb/sma
"""
import torch
import torch.nn as nn
import copy
from basicts.archs.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SimpleMovingAverage(nn.Module):
    def __init__(self, q: int, input_length: int, output_length: int):
        """simple moving average as prediction

        Args:
            q (int): sliding window size
            input_length (int): length of input history data
            output_length (int): length of prediction
        """
        super(SimpleMovingAverage, self).__init__()
        assert input_length >= q, "Error: window size > input data length"
        self.q = q
        self.output_length = output_length
        self.input_length  = input_length

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward of MA: https://github.com/doowb/sma
            forward([1, 2, 3, 4, 5, 6, 7, 8, 9]) | p=4;
            //=> [ '2.50', '3.50', '4.50', '5.50', '6.50', '7.50' ]
            //=>   │       │       │       │       │       └─(6+7+8+9)/4
            //=>   │       │       │       │       └─(5+6+7+8)/4
            //=>   │       │       │       └─(4+5+6+7)/4
            //=>   │       │       └─(3+4+5+6)/4
            //=>   │       └─(2+3+4+5)/4
            //=>   └─(1+2+3+4)/4

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: MA prediction
        """
        [B, L, N, C] = history_data.shape
        assert L == self.input_length, "error input data length"
        data_full = copy.copy(history_data)
        for i in range(self.output_length):
            data_in_window  = data_full[:, -self.q:, :, :]
            simple_avg      = torch.mean(data_in_window, dim=1)         # [B, N, C]
            data_full       = torch.cat([data_full, simple_avg.unsqueeze(1)], dim=1)
        prediction = data_full[:, -self.output_length:, :, :]
        return prediction


@ARCH_REGISTRY.register()
class AutoRegressive(nn.Module):
    def __init__(self, p: int, input_length: int, output_length: int):
        """Auto Regressive (AR) model

        Args:
            p (int): sliding window size
            input_length (int): length of input history data
            output_length (int): length of prediction
        """
        super(AutoRegressive, self).__init__()
        assert input_length >= p, "Error: window size > input data length"
        self.p = p
        self.output_length = output_length
        self.input_length  = input_length
        self.weight = nn.Parameter(torch.empty(p, 1))
        print("Notes: the weights of WMA model are unnormalized.")
        self.c = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.c)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward of autoregressive model: https://en.wikipedia.org/wiki/Autoregressive_model

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: MA prediction
        """
        [B, L, N, C] = history_data.shape
        assert L == self.input_length, "error input data length"
        data_full = copy.copy(history_data)
        for i in range(self.output_length):
            data_in_window  = data_full[:, -self.p:, :, :]                                      # [B, p, N, C]
            data_in_window  = data_in_window.permute(0, 2, 3, 1)                                # [B, N, C, p]
            weight_avg      = torch.matmul(data_in_window, self.weight).permute(0, 3, 1, 2)     # [B, 1, N, C]
            weight_avg      = weight_avg + self.c                                               # the noise term is omitted
            data_full       = torch.cat([data_full, weight_avg], dim=1)
        prediction = data_full[:, -self.output_length:, :, :]
        return prediction


@ARCH_REGISTRY.register()
class VectorAutoRegression(nn.Module):
    def __init__(self, p: int, input_length: int, output_length: int, num_time_series: int):
        """vector auto regressive model for multivariate time series forecasting

        Args:
            p (int): sliding window size
            input_length (int): length of input history data
            output_length (int): length of prediction
            num_time_series (int): number of time series
        """
        super(VectorAutoRegression, self).__init__()
        self.p = p
        self.output_length = output_length
        self.input_length  = input_length
        self.N = num_time_series
        self.weight = nn.Parameter(torch.empty(p, self.N, self.N))      # [p, N, N]
        print("Notes: the weights of VAR model are unnormalized.")
        self.c = nn.Parameter(torch.empty(self.N, 1))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.c)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward of VAR: https://en.wikipedia.org/wiki/Vector_autoregression

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: VAR prediction
        """
        [B, L, N, C] = history_data.shape
        assert L == self.input_length, "error input data length"
        data_full = copy.copy(history_data)
        for i in range(self.output_length):
            data_in_window  = data_full[:, -self.p:, :, :]                                      # [B, p, N, C]
            data_in_window  = data_in_window.permute(0, 3, 1, 2).unsqueeze(-1)                  # [B, C, p, N, 1]
            weighted_data   = torch.matmul(self.weight, data_in_window).squeeze(-1)             # [B, C, p, N]
            weight_avg      = torch.mean(weighted_data, dim=-2).permute(0, 2, 1).unsqueeze(1)   # [B, 1, N, C]
            weight_avg      = weight_avg + self.c                                               # error term is omitted
            data_full       = torch.cat([data_full, weight_avg], dim=1)
        prediction = data_full[:, -self.output_length:, :, :]
        return prediction


@ARCH_REGISTRY.register()
class ARIMA(nn.Module):
    def __init__(self):
        super(ARIMA, self).__init__()
        """TODO: ARIMA model requires unnormalized data to add N(0, 1) noise. 
        """
        pass

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
