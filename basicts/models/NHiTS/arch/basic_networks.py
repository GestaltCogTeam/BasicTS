from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or ("cubic" in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
        self,
        backcast_theta: torch.Tensor,
        forecast_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )  # , align_corners=True)
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


class NHiTSBlock(nn.Module):
    def __init__(self, context_length: int, prediction_length: int,
                output_size: int, n_theta: int, hidden_size: List[int],
                pooling_sizes: int, pooling_mode: str, basis: nn.Module,
                n_layers: int, batch_normalization: bool, dropout: float,
                activation: str):
        super().__init__()

        assert pooling_mode in ["max", "average"]

        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))

        self.context_length = context_length
        self.output_size = output_size
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation_list = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

        self.hidden_size = [self.context_length_pooled] + hidden_size

        assert activation in self.activation_list, f"{activation} is not in {self.activation_list}"
        activ = getattr(nn, activation)()

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i + 1]))

            if self.dropout > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout))

        output_layer = [
            nn.Linear(
                in_features=self.hidden_size[-1],
                out_features=context_length + n_theta * output_size,
            )
        ]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, encoder_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        # Pooling layer to downsample input
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        # Compute local projection weights and projection
        theta = self.layers(encoder_y)
        backcast_theta = theta[:, : self.context_length].reshape(-1, self.context_length)
        forecast_theta = theta[:, self.context_length:].reshape(-1, self.n_theta)
        backcast, forecast = self.basis(backcast_theta, forecast_theta)
        backcast = backcast.reshape(-1, 1, self.context_length).transpose(1, 2)
        forecast = forecast.reshape(-1, self.output_size, self.prediction_length).transpose(1, 2)

        return backcast, forecast
