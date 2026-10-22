"""Calendar covariates and convolutional embeddings for forecasting models.

Calendar inputs here are centered in [-0.5, 0.5). Unlike TimestampEmbedding,
continuous covariates use a learned linear projection with a bias.
"""

import math
import torch
from torch import nn
from .tst_embed import PositionEmbedding, TokenEmbedding


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        time_of_day_size,
        day_of_week_size,
        day_of_month_size,
        day_of_year_size,
        embed_type="fixed",
    ):
        super(TemporalEmbedding, self).__init__()

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.day_of_month_size = day_of_month_size
        self.day_of_year_size = day_of_year_size

        if time_of_day_size is not None:
            self.time_of_day_embed = Embed(time_of_day_size, d_model)
        if day_of_week_size is not None:
            self.day_of_week_embed = Embed(day_of_week_size, d_model)
        if day_of_month_size is not None:
            self.day_of_month_embed = Embed(day_of_month_size, d_model)
        if day_of_year_size is not None:
            self.day_of_year_embed = Embed(day_of_year_size, d_model)

    def forward(self, x):
        x = (
            x + 0.5
        )  # re-normalize to [0, 1) to get correct index. ref: arch_zoo/utils/xformer.py data_transformation_4_xformer
        temporal_embeddings = []
        if self.time_of_day_size is not None:
            time_of_day_x = self.time_of_day_embed(
                (x[:, :, 0] * self.time_of_day_size).long()
            )
            temporal_embeddings.append(time_of_day_x)
        if self.day_of_week_size is not None:
            day_of_week_x = self.day_of_week_embed(
                (x[:, :, 1] * self.day_of_week_size).long()
            )
            temporal_embeddings.append(day_of_week_x)
        if self.day_of_month_size is not None:
            day_of_month_x = self.day_of_month_embed(
                (x[:, :, 2] * self.day_of_month_size).long()
            )
            temporal_embeddings.append(day_of_month_x)
        if self.day_of_year_size is not None:
            day_of_year_x = self.day_of_year_embed(
                (x[:, :, 3] * self.day_of_year_size).long()
            )
            temporal_embeddings.append(day_of_year_x)

        if len(temporal_embeddings) == 0:
            return 0
        else:
            return sum(temporal_embeddings)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, num_time_features):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(num_time_features, d_model, bias=True)

    def forward(self, x):
        return self.embed(x)


class BiasedTokenEmbedding(TokenEmbedding):
    """TokenEmbedding variant retaining Pyraformer's learned convolution bias."""

    def __init__(self, num_features, hidden_size):
        nn.Module.__init__(self)
        self.tokenConv = nn.Conv1d(
            num_features,
            hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self._init_weights()


class CalendarFeatureEmbedding(nn.Module):
    """Value embedding with optional positions and centered calendar covariates.

    `use_timestamps=False` preserves Pyraformer's value/position-only path.
    Calendar parameters remain registered for checkpoint compatibility.
    """

    def __init__(
        self,
        num_features,
        hidden_size,
        timestamp_sizes=(24, 7, 31, 366),
        num_timestamps=4,
        continuous=False,
        fixed=False,
        use_pe=True,
        use_timestamps=True,
        value_bias=False,
        dropout=0.1,
    ):
        super().__init__()
        self.value_embedding = (BiasedTokenEmbedding if value_bias else TokenEmbedding)(
            num_features, hidden_size
        )
        self.position_embedding = PositionEmbedding(hidden_size)
        if continuous:
            self.temporal_embedding = TimeFeatureEmbedding(hidden_size, num_timestamps)
        else:
            self.temporal_embedding = TemporalEmbedding(
                hidden_size,
                *timestamp_sizes,
                embed_type="fixed" if fixed else "learned",
            )
        self.use_pe = use_pe
        self.use_timestamps = use_timestamps
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, timestamps=None):
        value = self.value_embedding(inputs)
        if self.use_pe:
            value = value + self.position_embedding(inputs)
        if self.use_timestamps and timestamps is not None:
            value = value + self.temporal_embedding(timestamps)
        return self.dropout(value)


class ConvolutionalPatchEmbedding(nn.Module):
    """Replicated-end patches projected by convolution across patch positions.

    This differs from the linear patch projection in tst_embed.PatchEmbedding.
    Input: [batch, features, length].
    """

    def __init__(self, hidden_size, patch_len, stride, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, hidden_size)
        self.position_embedding = PositionEmbedding(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        num_features = inputs.shape[1]
        patches = self.padding_patch_layer(inputs).unfold(
            -1, self.patch_len, self.stride
        )
        patches = patches.reshape(-1, patches.shape[-2], patches.shape[-1])
        return self.dropout(
            self.value_embedding(patches) + self.position_embedding(patches)
        ), num_features


class CausalFeatureEmbedding(nn.Module):
    """Causal convolutional value embedding used by ETSformer."""

    def __init__(self, num_features, hidden_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            num_features, hidden_size, kernel_size=3, padding=2, bias=False
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        embedded = self.conv(inputs.transpose(1, 2))[..., :-2]
        return self.dropout(embedded.transpose(1, 2))
