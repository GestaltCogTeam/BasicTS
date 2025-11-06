import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from basicts.modules.transformer import EncoderLayer


class LeddamEncoderLayer(EncoderLayer):
    """
    Leddam Transformer block.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, Any]],
        attn_norm: nn.Module
    ):
        super().__init__(self_attn, ffn_layer, layer_norm, "post")
        # Leddam uses batch norm for post-attention norm
        self.post_attn_norm = attn_norm


class LearnableDecomposition(nn.Module):
    """
    Learnable Decomposition module for Leddam.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # Define a shared convolution layers for all channels
        self.conv=nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, padding_mode="replicate")
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        x = x.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(x, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class AutoAttention(nn.Module):
    """
    Autoregressive Attention module for Leddam.
    """
    def __init__(self, hidden_size: int, period: int = 64, dropout: float = 0.1):
        """
        Initialize the Auto-Attention module.

        Args:
            d_model (int): The input and output dimension for queries, keys, and values.
        """
        super().__init__()
        self.period = period
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def auto_attention(self, inp):
        """
        Perform auto-attention mechanism on the input.

        Args:
            inp (torch.Tensor): Input data of shape [B, N, T], where B is the batch size,
                               N is the number of features, and T is the sequence length.
        Returns:
            output (torch.Tensor): Output after auto-attention.
        """
        # Separate query and key
        query = self.q_proj(inp[:, :, 0, :].unsqueeze(-2))  # Query
        keys = self.k_proj(inp)  # Keys
        values = self.v_proj(inp)  # Values

        # Calculate dot product
        attn_scores = torch.matmul(query, keys.transpose(-2, -1))

        # Normalize attention scores
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attn_scores, values)

        return output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the Auto-Attention module.

        Args:
            P (int): The period for autoregressive behavior.
            inp (torch.Tensor): Input data of shape [B, T, N], where B is the batch size,
                               T is the sequence length, and N is the number of features.

        Returns:
            output (torch.Tensor): Output after autoregressive self-attention.
        """

        num_shifts = self.hidden_size // self.period
        hidden_states_rolled = [torch.roll(hidden_states, shifts=i * self.period, dims=-1)
                                for i in range(num_shifts)]
        # Stack the concatenated sequences: [B, N, num_shifts, hidden_size]
        hidden_states = torch.stack(hidden_states_rolled, dim=-1).transpose(-2, -1)

        query = self.q_proj(hidden_states[:, :, 0:1, :])
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        output = self.out_proj(context.squeeze(-2))

        if not output_attentions:
            attn_weights = None

        return output, attn_weights, None
