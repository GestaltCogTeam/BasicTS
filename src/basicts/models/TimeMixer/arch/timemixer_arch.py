from typing import Optional

import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.embed import FeatureEmbedding
from basicts.modules.norm import RevIN

from ..config.timemixer_config import TimeMixerConfig
from .mixing_layers import PastDecomposableMixing


class TimeMixerBackBone(nn.Module):
    """
    Paper: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
    Official Code: https://github.com/kwuking/TimeMixer
    Link: https://arxiv.org/html/2405.14616v1
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: TimeMixerConfig):
        super().__init__()
        self.down_sampling_layers = config.down_sampling_layers
        self.down_sampling_window = config.down_sampling_window
        self.channel_independence = config.channel_independence
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(config) for _ in range(config.num_layers)])
        self.decomp_layer = MovingAverageDecomposition(config.moving_avg)
        if config.down_sampling_method == "max":
            self.down_pooling = nn.MaxPool1d(self.down_sampling_window)
        elif config.down_sampling_method == "avg":
            self.down_pooling = nn.AvgPool1d(self.down_sampling_window)
        elif config.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            self.down_pooling = nn.Conv1d(
                config.num_features,
                config.num_features,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False)
        embed_channels = 1 if self.channel_independence else config.num_features
        self.enc_embedding = FeatureEmbedding(embed_channels,
            config.hidden_size,
            use_timestamps=config.use_timestamps,
            timestamp_sizes=config.timestamp_sizes,
            dropout=config.dropout)
        self.norm_layers = nn.ModuleList([RevIN(config.num_features, affine=True)
                                                for _ in range(self.down_sampling_layers + 1)])

    def _decomposition(
            self,
            x_list: list[torch.Tensor]
            ) -> tuple[list[torch.Tensor], Optional[list[torch.Tensor]]]:
        if self.channel_independence:
            return x_list, None
        else:
            seasonal_list = []
            trend_list = []
            for x in x_list:
                seasonal, trend = self.decomp_layer(x)
                seasonal_list.append(seasonal)
                trend_list.append(trend)
            return seasonal_list, trend_list

    def _prepare_multi_scale_inputs(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None
            ) -> tuple[list[torch.Tensor], Optional[list[torch.Tensor]]]:

        multi_scale_inputs = [inputs]
        multi_scale_timestamps = [inputs_timestamps] if inputs_timestamps is not None else None
        sample = inputs.permute(0, 2, 1) # [batch_size, num_features, seq_len]
        sample_ts = inputs_timestamps.permute(0, 2, 1) if inputs_timestamps is not None else None

        for _ in range(self.down_sampling_layers):
            down_sampled = self.down_pooling(sample)
            multi_scale_inputs.append(down_sampled.permute(0, 2, 1))
            sample = down_sampled

            if inputs_timestamps is not None:
                multi_scale_timestamps.append(
                    sample_ts[:, :, ::self.down_sampling_window].permute(0, 2, 1))
                sample_ts = sample_ts[:, :, ::self.down_sampling_window]

        return multi_scale_inputs, multi_scale_timestamps

    def forward(self,
                inputs: torch.Tensor,
                inputs_timestamps: Optional[torch.Tensor] = None,
                multi_scale_norm: bool = False,
                decomp: bool = False,
                ) -> tuple[list[torch.Tensor], Optional[list[torch.Tensor]]]:

        x_list, x_ts_list = self._prepare_multi_scale_inputs(inputs, inputs_timestamps)
        num_scales = len(x_list)

        for i in range(num_scales):
            _, input_len, num_features = x_list[i].size()
            if multi_scale_norm:
                x_list[i] = self.norm_layers[i](x_list[i], "norm")
            if self.channel_independence:
                x_list[i] = x_list[i].transpose(1, 2).reshape(-1, input_len, 1)
            if x_ts_list is not None:
                x_ts_list[i] = x_ts_list[i].repeat(num_features, 1, 1)

        # decomposition for forecasting task
        if decomp:
            if self.channel_independence:
                res_list = None
            else:
                res_list = []
                for x in x_list:
                    x, res = self.decomp_layer(x)
                    res_list.append(res)

        # embedding
        hidden_states_list = []
        for i in range(num_scales):
            hidden_states = self.enc_embedding(x_list[i], x_ts_list[i])
            hidden_states_list.append(hidden_states)

        # Past Decomposable Mixing as encoder for past
        for layer in self.pdm_blocks:
            hidden_states_list = layer(hidden_states_list)

        return hidden_states_list, res_list


class TimeMixerForForecasting(nn.Module):

    """
    TimeMixer for time series forecasting.
    """

    def __init__(self, config: TimeMixerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.down_sampling_layers = config.down_sampling_layers
        self.down_sampling_window = config.down_sampling_window
        self.backbone = TimeMixerBackBone(config)
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    config.input_len // (self.down_sampling_window ** i),
                    config.output_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        if config.channel_independence:
            self.projection = nn.Linear(config.hidden_size, 1)
        else:
            self.projection = nn.Linear(config.hidden_size, config.num_features)
            self.out_res_layers = nn.ModuleList(
                [
                    nn.Linear(
                        config.input_len // (self.down_sampling_window ** i),
                        config.input_len // (self.down_sampling_window ** i),
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )
            self.regression_layers = torch.nn.ModuleList(
                [
                    nn.Linear(
                        config.input_len // (self.down_sampling_window ** i),
                        config.output_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

    def forward(self,
                inputs: torch.Tensor,
                inputs_timestamps: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        hidden_states_list, res_list = self.backbone(
            inputs, inputs_timestamps, multi_scale_norm=True, decomp=True)
        prediction_list = []
        for i, hidden_states in enumerate(hidden_states_list):
            hidden_states = self.predict_layers[i](
                hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
            hidden_states = self.projection(hidden_states)
            if res_list is None: # channel indepenent
                hidden_states = hidden_states.reshape(
                    -1, self.num_features, self.output_len).permute(0, 2, 1)
            else:
                res = self.out_res_layers[i](res_list[i].permute(0, 2, 1))
                res = self.regression_layers[i](res).permute(0, 2, 1)
                hidden_states = hidden_states + res
            prediction_list.append(hidden_states)
        prediction = torch.stack(prediction_list, dim=-1).sum(dim=-1)
        prediction = self.backbone.norm_layers[0](prediction, "denorm")
        return prediction
