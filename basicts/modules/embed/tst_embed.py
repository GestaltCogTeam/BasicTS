from typing import Optional, Sequence

import torch
from torch import nn


class TimestampEmbedding(nn.Module):
    """
    TimestampEmbedding layer is used to embed the timestamps to hidden dimension, \
    i.e., [batch_size, seq_len, num_timestamps] -> [batch_size, seq_len, hidden_size].
    """
    def __init__(
            self,
            hidden_size: int,
            timestamp_sizes: Sequence[int]):
        super().__init__()

        self.timestamp_sizes = timestamp_sizes
        self.embeds = nn.ModuleList(
            [nn.Embedding(timestamp_size, hidden_size) for timestamp_size in timestamp_sizes]
        )


    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimestampEmbedding layer.

        Args:
            timestamps (torch.Tensor): Input tensor of timestamps of shape [batch_size, seq_len, num_timestamps].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        temporal_embeddings = []
        for i, embed in enumerate(self.embeds):
            temporal_embeddings.append(embed((timestamps[..., i] * self.timestamp_sizes[i]).long()))
        return sum(temporal_embeddings)


class FeatureEmbedding(nn.Module):
    """
    FeatureEmbedding layer is used to embed the time series from the feature dimension to hidden dimension, \
    i.e., [batch_size, seq_len, num_features] -> [batch_size, seq_len, hidden_size].
    """
    def __init__(
            self,
            num_features: int,
            hidden_size: int,
            use_timestamps: bool = False,
            timestamp_sizes: Optional[Sequence[int]] = None,
            dropout: float = 0.1):

        super().__init__()
        self.value_embedding = nn.Linear(num_features, hidden_size)
        self.use_timestamps = use_timestamps
        if use_timestamps:
            self.timestamp_embedding = TimestampEmbedding(hidden_size, *timestamp_sizes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the FeatureEmbedding layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features].
            inputs_timestamps (torch.Tensor): Optional tensor of timestamps of shape [batch_size, seq_len, num_timestamps]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        x = self.value_embedding(inputs)
        if self.use_timestamps and inputs_timestamps is not None:
            x += self.timestamp_embedding(inputs_timestamps)
        return self.dropout(x)

class SequenceEmbedding(nn.Module):
    """
    SequenceEmbedding layer is used to embed the time series from the temporal dimension to hidden dimension, \
    i.e., [batch_size, seq_len, num_features] -> [batch_size, num_features, hidden_size].
    """
    def __init__(
            self,
            seq_len: int,
            hidden_size: int,
            dropout: float = 0.1):

        super().__init__()
        self.value_embedding = nn.Linear(seq_len, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the SequenceEmbedding layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features].
            inputs_timestamps (torch.Tensor): Optional tensor of timestamps of shape [batch_size, seq_len, num_timestamps]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """

        inputs = inputs.transpose(1,2)
        if inputs_timestamps is None:
            embedded = self.value_embedding(inputs)
        else:
            # the potential to take timestamps as tokens
            embedded = self.value_embedding(torch.cat([inputs, inputs_timestamps.transpose(1,2)], dim=1))
        # embedded: [batch_size, num_features (+ num_timestamps), hidden_size]
        return self.dropout(embedded)
