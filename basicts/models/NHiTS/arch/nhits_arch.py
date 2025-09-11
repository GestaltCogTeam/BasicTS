# modified from pytorch_forecasting.NHiTS
import torch
import torch.nn as nn
from functools import partial

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
            pass  # torch.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1 < 0, f"Initialization {initialization} not found"

class NHiTS(nn.Module):
    """
    Paper: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
    Link: https://arxiv.org/abs/2201.12886
    Official code: https://github.com/cchallu/n-hits
    Venue: AAAI 2023
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, context_length: int, prediction_length: int, output_size: int,
                n_blocks: list, n_layers: list, hidden_size: list, pooling_sizes: list,
                downsample_frequencies: list, pooling_mode: str="max", interpolation_mode: str="linear",
                dropout: float=0.0, activation: str="ReLU", initialization: str="lecun_normal",
                batch_normalization: bool=False, shared_weights: bool=False, naive_level: bool=True):
        super().__init__()
    
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

    def create_stack(self, n_blocks, context_length, prediction_length, output_size,
                    n_layers, hidden_size, pooling_sizes, downsample_frequencies,
                    pooling_mode, interpolation_mode, batch_normalization, dropout,
                    activation, shared_weights, initialization):
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
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

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list
 
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Forward pass of the NHiTS model.

        Args:
            history_data (torch.Tensor): History data of shape [B, L, N, C]

        Returns:
            torch.Tensor: Predictions of shape [B, L, N, C]
        """
        B, L, N, C = history_data.shape
        history_data = history_data[..., [0]].transpose(1, 2)   # [B, N, L, 1]
        history_data = history_data.reshape(B*N, L, 1)

        residuals = history_data
        level = history_data[:, -1:].repeat(1, self.prediction_length, 1)  # Level with Naive1
        forecast_level = level.repeat_interleave(torch.tensor(self.output_size, device=level.device), dim=2)

        # level with last available observation
        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [history_data[:, -1:].repeat(1, self.context_length, 1)]
            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)

        # forecast by block
        for block in self.blocks:
            block_backcast, block_forecast = block(encoder_y=residuals)
            residuals = (residuals - block_backcast)
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)

        # (n_batch, n_t, n_outputs, n_blocks)
        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        backcast = residuals
        # return forecast, backcast, block_forecasts, block_backcasts
        forecast = forecast.reshape(B, N, self.prediction_length, self.output_size).transpose(1, 2) # [B, L, N, C]
        return forecast
