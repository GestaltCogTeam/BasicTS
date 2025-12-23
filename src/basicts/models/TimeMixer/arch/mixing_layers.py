import torch
from torch import nn

from basicts.modules.decomposition import (DFTDecomposition,
                                           MovingAverageDecomposition)
from basicts.modules.mlps import MLPLayer

from ..config.timemixer_config import TimeMixerConfig


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, config: TimeMixerConfig):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                MLPLayer(
                    config.input_len // (config.down_sampling_window ** i),
                    config.input_len // (config.down_sampling_window ** (i + 1)),
                    config.input_len // (config.down_sampling_window ** (i + 1)),
                    hidden_act=config.hidden_act,
                )
                for i in range(config.down_sampling_layers)
            ]
        )

    def forward(self, seasonal_list: list[torch.Tensor]) -> list[torch.Tensor]:

        # mixing high->low
        out_high = seasonal_list[0]
        out_low = seasonal_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(seasonal_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(seasonal_list) - 1:
                out_low = seasonal_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, config: TimeMixerConfig):
        super().__init__()

        self.up_sampling_layers = nn.ModuleList(
            [
                MLPLayer(
                    config.input_len // (config.down_sampling_window ** (i + 1)),
                    config.input_len // (config.down_sampling_window ** i),
                    config.input_len // (config.down_sampling_window ** i),
                    hidden_act=config.hidden_act,
                )
                for i in reversed(range(config.down_sampling_layers))
            ])

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    Past decomposable mixing layer.
    """

    def __init__(self, config: TimeMixerConfig):
        super().__init__()
        self.channel_independence = config.channel_independence
        self.decomp_method = config.decomp_method
        if self.decomp_method == "moving_avg":
            self.decompsition = MovingAverageDecomposition(config.moving_avg)
        elif self.decomp_method == "dft_decomp":
            self.decompsition = DFTDecomposition(config.top_k)
        else:
            raise ValueError(f"decompsition type {self.decomp_method} is not supported.")

        if not self.channel_independence:
            self.cross_layer = MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act)

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(config)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(config)

        self.out_cross_layer = MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act)

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        length_list = []
        for x in x_list:
            length_list.append(x.size(1))

        # Decompose to obtain the season and trend
        seasonal_list, trend_list = [], []
        for x in x_list:
            seasonal, trend = self.decompsition(x)
            if not self.channel_independence:
                seasonal = self.cross_layer(seasonal)
                trend = self.cross_layer(trend)
            seasonal_list.append(seasonal.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        seasonal_list = self.mixing_multi_scale_season(seasonal_list)
        # top-down trend mixing
        trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for x, seasonal, trend in zip(x_list, seasonal_list, trend_list):
            out = seasonal + trend
            if self.channel_independence:
                out = x + self.out_cross_layer(out)
            out_list.append(out)
        return out_list
