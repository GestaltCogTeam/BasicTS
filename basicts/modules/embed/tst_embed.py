import math
from typing import Literal, Optional, Sequence

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """
    PositionEmbedding layer is used to embed the position of the time series to hidden dimension, \
    i.e., [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size].
    """

    def __init__(self, hidden_size: int, max_len: int = 5000):
        """
        Args:
            hidden_size (int): size of hidden dimension.
            max_len (int, optional): maximum length of the time series. Defaults to 5000.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) \
                             * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Position embedding.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding layer is used to embed the time series from the feature dimension to hidden dimension, \
    i.e., [batch_size, num_features, seq_len] -> [batch_size, seq_len, hidden_size].
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(num_features, hidden_size, kernel_size=3,
                                   padding=padding, padding_mode="circular", bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.tokenConv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenConv(x.transpose(1, 2)) # [batch_size, hidden_size, seq_len]
        return x.transpose(1, 2)


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
            embed_type: Literal["token", "linear"] = "token",
            use_timestamps: bool = False,
            timestamp_sizes: Optional[Sequence[int]] = None,
            use_pe: bool = False,
            dropout: float = 0.1):

        super().__init__()
        if embed_type == "token":
            self.value_embedding = TokenEmbedding(num_features, hidden_size)
        elif embed_type == "linear":
            self.value_embedding = nn.Linear(num_features, hidden_size)
        else:
            raise ValueError(f"Unknown embed_type {embed_type}")
        self.use_timestamps = use_timestamps
        if use_timestamps:
            self.timestamp_embedding = TimestampEmbedding(hidden_size, timestamp_sizes)
        if use_pe:
            self.position_embedding = PositionEmbedding(hidden_size)
        else:
            self.position_embedding = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the FeatureEmbedding layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features].
            inputs_timestamps (torch.Tensor): Optional tensor of timestamps of shape [batch_size, seq_len, num_timestamps]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        embedded = self.value_embedding(inputs)
        if self.use_timestamps and inputs_timestamps is not None:
            embedded += self.timestamp_embedding(inputs_timestamps)
        if self.position_embedding is not None:
            embedded += self.position_embedding(embedded)
        return self.dropout(embedded)

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


class PatchEmbedding(nn.Module):
    """
    PatchEmbedding layer is used to embed the time series from the patch dimension to hidden dimension, \
    i.e., [batch_size, seq_len, num_features] -> [batch_size * num_features, num_patches, hidden_size].
    """

    def __init__(
            self,
            hidden_size: int,
            patch_len: int = 16,
            stride: int = 8,
            padding: Optional[tuple[int, int]] = None,
            dropout: float = 0.1):

        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        self.padding_layer = nn.ReplicationPad1d(padding) if padding is not None else None

        self.value_embedding = nn.Linear(patch_len, hidden_size)
        self.position_embedding = PositionEmbedding(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * num_features, num_patches, hidden_size].
        """
        inputs = inputs.transpose(1, 2)
        if self.padding_layer is not None:
            # padding: # [batch_size, num_features, seq_len + padding]
            inputs = self.padding_layer(inputs)
        # patching: # [batch_size, num_features, num_patches, patch_len]
        patches = inputs.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # embedding: # [batch_size * num_features, num_patches, hidden_size]
        patches = patches.reshape(-1, patches.size(2), self.patch_len)
        embedded = self.value_embedding(patches) + self.position_embedding(patches)
        return self.dropout(embedded)
