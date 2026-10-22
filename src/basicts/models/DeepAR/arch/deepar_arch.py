"""DeepAR with native BasicTS tensors and a shared Gaussian likelihood head."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicts.modules.embed.time_features import calendar_features
from basicts.modules.probabilistic import Gaussian

from ..config.deepar_config import DeepARConfig


class DeepAR(nn.Module):
    """
    Paper: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Link: https://doi.org/10.1016/j.ijforecast.2019.07.001
    Ref Official Code: https://github.com/awslabs/gluonts
    Venue: International Journal of Forecasting 2020
    Task: Probabilistic Time Series Forecasting

    Note:
        This BasicTS adaptation uses a Gaussian likelihood and autoregressive sampling.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: DeepARConfig):
        super().__init__()
        self.output_len = config.output_len
        self.num_timestamps = config.num_timestamps
        self.use_series_id = config.use_series_id
        self.input_embed = nn.Linear(1, config.embedding_size)
        id_size = config.id_embedding_size if self.use_series_id else 0
        if self.use_series_id:
            self.id_feat = nn.Parameter(torch.empty(config.num_features, id_size))
            nn.init.xavier_uniform_(self.id_feat)
        self.encoder = nn.LSTM(
            config.embedding_size + self.num_timestamps + id_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
        )
        self.likelihood_layer = Gaussian(config.hidden_size, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
        targets_timestamps: Optional[torch.Tensor] = None,
        train: Optional[bool] = None,
    ):
        training = self.training if train is None else train
        batch, input_len, features = inputs.shape
        history_marks = calendar_features(
            inputs_timestamps, inputs, input_len, self.num_timestamps, center=False
        )
        future_marks = calendar_features(
            targets_timestamps,
            inputs,
            self.output_len,
            self.num_timestamps,
            center=False,
        )
        future = (
            targets
            if training and targets is not None
            else inputs.new_zeros(batch, self.output_len, features)
        )
        values = torch.cat([inputs, future], dim=1)
        marks = torch.cat([history_marks, future_marks], dim=1)
        predictions, means, scales = [], [], []
        next_value = None
        hidden = None
        for time in range(1, input_len + self.output_len):
            if time <= input_len or training:
                next_value = values[:, time - 1 : time, :]
            embedded = self.input_embed(next_value.unsqueeze(-1))
            covariates = marks[:, time : time + 1, None, :].expand(-1, -1, features, -1)
            parts = [embedded, covariates]
            if self.use_series_id:
                parts.append(self.id_feat[None, None].expand(batch, 1, -1, -1))
            encoded = (
                torch.cat(parts, dim=-1)
                .transpose(1, 2)
                .reshape(batch * features, 1, -1)
            )
            _, hidden = (
                self.encoder(encoded)
                if hidden is None
                else self.encoder(encoded, hidden)
            )
            mean, scale = self.likelihood_layer(F.relu(hidden[0][-1]))
            sample = (
                torch.distributions.Normal(mean.squeeze(1), scale.squeeze(1))
                .sample([1])
                .squeeze(0)
            )
            next_value = sample.reshape(batch, 1, features)
            predictions.append(next_value)
            means.append(mean.reshape(batch, 1, features))
            scales.append(scale.reshape(batch, 1, features))
        mu = torch.cat(means, dim=1)[:, -self.output_len :]
        sigma = torch.cat(scales, dim=1)[:, -self.output_len :]
        prediction = (
            mu if training else torch.cat(predictions, dim=1)[:, -self.output_len :]
        )
        result = {"prediction": prediction, "mus": mu, "sigmas": sigma}
        if training and targets is not None:
            result["loss"] = (
                -torch.distributions.Normal(mu, sigma).log_prob(targets).mean()
            )
        return result
