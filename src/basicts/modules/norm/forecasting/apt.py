"""APT timestamp affine module, adapted from blisky-li/APT at 98a4c9c.

Timestamps follow BasicTS's [0, 1) calendar fractions. The reference code
subtracts 0.5 before APT and adds it back inside; those operations cancel here.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class APT(nn.Module):
    """
    Paper: APT: Affine Prototype-Timestamp for Time Series Forecasting Under Distribution Shift
    Link: https://ojs.aaai.org/index.php/AAAI/article/view/38542
    Ref Official Code: https://github.com/blisky-li/APT
    Venue: AAAI 2026
    Task: Timestamp-conditioned Affine Adaptation for Time Series Forecasting
    """

    def __init__(self, num_features, timestamp_features=("time_of_day", "day_of_week"),
                 timestamp_dim=20, timestamp_hidden=32, num_prototypes=30,
                 top_k=3, independent=False, time_of_day_size=24,
                 day_of_week_size=7, day_of_month_size=31, day_of_year_size=366):
        super().__init__()
        self.timestamp_features = tuple(timestamp_features)
        self.independent = independent
        self.num_timeseries = num_features
        self.topk = top_k
        self.sizes = (time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size)
        self.names = ("time_of_day", "day_of_week", "day_of_month", "day_of_year")
        if not self.timestamp_features or len(set(self.timestamp_features)) != len(self.timestamp_features):
            raise ValueError("APT requires distinct timestamp_features.")
        if set(self.timestamp_features) - set(self.names):
            raise ValueError(f"Unknown APT timestamp feature: {self.timestamp_features}")
        if not 1 <= top_k <= num_prototypes:
            raise ValueError("APT requires 1 <= top_k <= num_prototypes.")
        # Keep the reference's parameter names and initialization order for
        # source checkpoint/numerical comparisons, including inactive gates.
        for name, short, size in zip(self.names, ("tod", "dow", "dom", "doy"), self.sizes):
            embedding = nn.Embedding(size, timestamp_dim) if name in self.timestamp_features else None
            if embedding is not None:
                if short in {"tod", "dow"}:
                    nn.init.xavier_uniform_(embedding.weight)
                else:
                    nn.init.normal_(embedding.weight)
            setattr(self, short, embedding)
        self.times_w = nn.Sequential(nn.Linear(timestamp_dim, timestamp_hidden), nn.GELU(), nn.Linear(timestamp_hidden, 1))
        self.times_b = nn.Sequential(nn.Linear(timestamp_dim, timestamp_hidden), nn.GELU(), nn.Linear(timestamp_hidden, 1))
        self.times_w.apply(_init_xavier)
        self.times_b.apply(_init_xavier)
        self.prototype_lib = nn.Parameter(torch.empty(num_prototypes, timestamp_dim))
        nn.init.xavier_uniform_(self.prototype_lib)
        self.shared_prototype = nn.Parameter(torch.empty(1, timestamp_dim))
        nn.init.xavier_uniform_(self.shared_prototype)
        self.layer_norm1 = nn.LayerNorm(timestamp_dim)
        self.layer_norm2 = nn.LayerNorm(timestamp_dim)
        self.gate_layer = nn.Linear(timestamp_dim, 1)
        self.gate_layer2 = nn.Linear(timestamp_dim, 1)
        self.weight_alpha = nn.Parameter(torch.ones(len(self.timestamp_features)))
        self.identify_embedding = nn.Parameter(torch.empty(num_features, timestamp_dim))
        nn.init.xavier_uniform_(self.identify_embedding)
        self.register_buffer("usage_count", torch.zeros(num_prototypes), persistent=False)

    def _match(self, query, train):
        similarity = query @ self.prototype_lib.T
        values, indices = similarity.topk(self.topk, dim=-1)
        weights = torch.zeros_like(similarity).scatter(-1, indices, values.softmax(-1))
        if train:
            # Retain the differentiable reference load for the balance loss.
            self.usage_count = weights.sum(dim=tuple(range(weights.ndim - 1)))
        return weights @ self.prototype_lib

    def forward(self, inputs_timestamps, targets_timestamps, train=None):
        train = self.training if train is None else train
        if inputs_timestamps is None or targets_timestamps is None:
            raise ValueError("APT requires both inputs_timestamps and targets_timestamps.")
        start, end = [], []
        for index, short in enumerate(("tod", "dow", "dom", "doy")):
            embedding = getattr(self, short)
            if embedding is None:
                continue
            for stamps in (inputs_timestamps, targets_timestamps):
                if stamps.ndim != 3 or stamps.shape[-1] <= index:
                    raise ValueError(f"APT requires timestamp column {index} ({self.names[index]}).")
                if not torch.isfinite(stamps[..., index]).all() or torch.any((stamps[..., index] < 0) | (stamps[..., index] >= 1)):
                    raise ValueError("APT timestamps must be finite BasicTS calendar fractions in [0, 1).")
            size = self.sizes[index]
            start_index = (inputs_timestamps[:, 0, index] * size).round().long().clamp(0, size - 1)
            end_index = (targets_timestamps[:, -1, index] * size).round().long().clamp(0, size - 1)
            start.append(embedding(start_index))
            end.append(embedding(end_index))
        alpha = self.weight_alpha.softmax(0)
        query_start = F.gelu(torch.einsum("bnd,n->bd", torch.stack(start, 1), alpha))
        query_end = F.gelu(torch.einsum("bnd,n->bd", torch.stack(end, 1), alpha))
        if self.independent:
            query_start = query_start[:, None, :].expand(-1, self.num_timeseries, -1)
            query_end = query_end[:, None, :].expand(-1, self.num_timeseries, -1)
        proto_start = self._match(query_start, train)
        proto_end = self._match(query_end, train)
        if self.independent:
            proto_start = (proto_start + self.identify_embedding) / math.sqrt(2)
            proto_end = (proto_end + self.identify_embedding) / math.sqrt(2)
        combined = (F.gelu(proto_start) + F.gelu(proto_end)) / math.sqrt(2)
        weight, bias = self.times_w(combined), self.times_b(combined)
        return (weight.transpose(1, 2), bias.transpose(1, 2)) if self.independent else (weight.unsqueeze(-1), bias.unsqueeze(-1))

    def get_combined_embeddings(self):
        return [getattr(self, short).weight for short in ("tod", "dow", "dom", "doy") if getattr(self, short) is not None] + [self.prototype_lib]

    def get_load(self):
        return self.usage_count
