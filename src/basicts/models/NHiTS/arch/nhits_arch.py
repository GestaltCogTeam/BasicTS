"""NHiTS forecasting architecture."""

from functools import partial

import torch
import torch.nn as nn

from ..config.nhits_config import NHiTSConfig
from .basic_networks import IdentityBasis, NHiTSBlock


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == "orthogonal":
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


class NHiTS(nn.Module):
    """
    Paper: NHITS: Neural Hierarchical Interpolation for Time Series Forecasting
    Link: https://ojs.aaai.org/index.php/AAAI/article/view/25854
    Ref Official Code: https://github.com/cchallu/n-hits
    Venue: AAAI 2023
    Task: Long-term Time Series Forecasting

    Note:
        The authors also maintain NHITS at https://github.com/Nixtla/neuralforecast.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: NHiTSConfig):
        super().__init__()
        context_length = config.input_len
        prediction_length = config.output_len
        output_size = config.output_size
        n_blocks = config.n_blocks
        n_layers = config.n_layers
        hidden_size = config.hidden_sizes
        pooling_sizes = config.pooling_sizes
        downsample_frequencies = config.downsample_frequencies
        pooling_mode = config.pooling_mode
        interpolation_mode = config.interpolation_mode
        dropout = config.dropout
        activation = config.hidden_act
        initialization = config.initialization
        batch_normalization = config.batch_normalization
        shared_weights = config.shared_weights
        naive_level = config.naive_level
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_size = output_size
        self.naive_level = naive_level
        blocks = self.create_stack(
            n_blocks=n_blocks,
            context_length=context_length,
            prediction_length=prediction_length,
            output_size=output_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def create_stack(
        self,
        n_blocks,
        context_length,
        prediction_length,
        output_size,
        n_layers,
        hidden_size,
        pooling_sizes,
        downsample_frequencies,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout,
        activation,
        shared_weights,
        initialization,
    ):
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):
                if len(block_list) == 0 and batch_normalization:
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)
                    basis = IdentityBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                        interpolation_mode=interpolation_mode,
                    )
                    nbeats_block = NHiTSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                    )
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, inputs) -> torch.Tensor:
        (B, L, N) = inputs.shape
        history_data = inputs.transpose(1, 2).reshape(B * N, L, 1)
        residuals = history_data
        level = history_data[:, -1:].repeat(1, self.prediction_length, 1)
        forecast_level = level.repeat_interleave(
            torch.tensor(self.output_size, device=level.device), dim=2
        )
        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [history_data[:, -1:].repeat(1, self.context_length, 1)]
            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)
        for block in self.blocks:
            (block_backcast, block_forecast) = block(encoder_y=residuals)
            residuals = residuals - block_backcast
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)
        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        backcast = residuals
        forecast = forecast.reshape(
            B, N, self.prediction_length, self.output_size
        ).transpose(1, 2)
        return forecast.squeeze(-1)
