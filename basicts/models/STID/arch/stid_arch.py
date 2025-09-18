import torch
from torch import nn

from basicts.modules import ResMLPLayer

from ..config.stid_config import STIDConfig


class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    Venue: CIKM 2022
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, config: STIDConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.input_hidden_size = config.input_hidden_size

        self.if_spatial = config.if_spatial
        self.if_time_in_day = config.if_time_in_day
        self.if_day_in_week = config.if_day_in_week
        self.num_time_in_day = config.num_time_in_day
        self.num_day_in_week = config.num_day_in_week

        # spatial embeddings
        if self.if_spatial:
            self.spatial_emb = nn.Parameter(
                torch.empty(config.num_features, config.spatial_hidden_size))
            nn.init.xavier_uniform_(self.spatial_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(config.num_time_in_day, config.tid_hidden_size))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(config.num_day_in_week, config.diw_hidden_size))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Linear(self.input_len, self.input_hidden_size)

        # encoding
        self.hidden_size = self.input_hidden_size + config.spatial_hidden_size * int(self.if_spatial) + \
            config.tid_hidden_size * int(self.if_time_in_day) + config.diw_hidden_size * int(self.if_day_in_week)
        self.intermediate_size = config.intermediate_size if config.intermediate_size is not None else self.hidden_size
        self.encoder = nn.Sequential(
            *[ResMLPLayer(self.hidden_size, self.intermediate_size, config.hidden_act) for _ in range(config.num_layers)])

        # regression layer
        self.regression_layer = nn.Linear(self.hidden_size, self.output_len)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            inputs (torch.Tensor): history data with shape [batch_size, input_len, num_features]
            inputs_timestamps (torch.Tensor): history timestamps with shape [batch_size, input_len, num_timestamps]

        Returns:
            torch.Tensor: prediction with shape [batch_size, output_len, num_features]
        """

        # In the datasets used in STID, timestamps is normalized to [0, 1].
        # We multiply it by `num_time_in_day` and `num_day_in_week`` to get the index.
        time_in_day_emb = self.time_in_day_emb[
            (inputs_timestamps[:, -1, 0] * self.num_time_in_day).type(torch.LongTensor)
            ] if self.if_time_in_day else None
        day_in_week_emb = self.day_in_week_emb[
            (inputs_timestamps[:, -1, 1] * self.num_day_in_week).type(torch.LongTensor)
            ] if self.if_day_in_week else None

        # time series embedding
        inputs = inputs.transpose(1, 2) # [batch_size, num_features, input_len]
        time_series_emb = self.time_series_emb_layer(inputs) # [batch_size, num_features, input_hidden_size]
        emb = [time_series_emb]

        # spatial embedding: [batch_size, num_features, spatial_hidden_size]
        if self.if_spatial:
            emb.append(self.spatial_emb.unsqueeze(0).expand(inputs.shape[0], -1, -1))
        # temporal embeddings: [batch_size, num_features, tid(diw)_hidden_size]
        if time_in_day_emb is not None:
            emb.append(time_in_day_emb.unsqueeze(1).expand(-1, inputs.shape[1], -1))
        if day_in_week_emb is not None:
            emb.append(day_in_week_emb.unsqueeze(1).expand(-1, inputs.shape[1], -1))

        # concate all embeddings: [batch_size, num_features, hidden_size]
        hidden = torch.cat(emb, dim=-1)

        # encoding: [batch_size, num_features, hidden_size]
        hidden = self.encoder(hidden)

        # regression: [batch_size, output_len, num_features]
        prediction = self.regression_layer(hidden).transpose(1, 2)

        return prediction
