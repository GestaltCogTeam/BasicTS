from typing import Iterable, Optional

import torch
from einops import rearrange, repeat
from torch import nn

from basicts.modules.mlps import MLPLayer
from basicts.modules.transformer import MultiHeadAttention


class PatchMergingLayer(nn.Module):
    """
    Patch Merging Layer.
    The adjacent `win_size` segments in each dimension will be merged into one patch \
    to get representation of a coarser scale
    """
    def __init__(self, hidden_size: int, win_size: int):
        super().__init__()
        self.win_size = win_size
        self.linear = nn.Linear(win_size * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(win_size * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size * num_features, num_patches, hidden_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * num_features, num // win_size, hidden_size]
        """
        num_patches = x.shape[1]
        pad_len = (self.win_size - num_patches % self.win_size) % self.win_size
        if pad_len > 0:
            x = torch.cat([x, x[:, -pad_len:, :]], dim=1)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, i::self.win_size, :])
        # [batch_size * num_features, num_patches // win_size, win_size * hidden_size]
        x = torch.cat(seg_to_merge, dim=-1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class CrossformerEncoderLayer(nn.Module):
    """
    Crossformer Encoder Layer
    """
    def __init__(
            self,
            num_features: int,
            hidden_size: int,
            n_heads: int,
            intermediate_size: int,
            num_patches: int,
            win_size: int = 1,
            factor: int = 10,
            hidden_act: str = "gelu",
            dropout: float =0.1):
        super().__init__()

        self.num_features = num_features
        self.merge_layer = PatchMergingLayer(
            hidden_size, win_size) if win_size > 1 else nn.Identity()
        self.time_attention = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.dim_sender = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.dim_receiver = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.router = nn.Parameter(torch.randn(num_patches, factor, hidden_size))

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)

        self.ffn1 = MLPLayer(
            hidden_size, intermediate_size, hidden_act=hidden_act, dropout=dropout)
        self.ffn2 = MLPLayer(
            hidden_size, intermediate_size, hidden_act=hidden_act, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        """
        Crossformer Encoder Layer

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size * num_features, num_patches, hidden_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * num_features, num_patches, hidden_size]
        """

        batch_size = hidden_states.shape[0] // self.num_features
        # Merge Patches
        hidden_states = self.merge_layer(hidden_states)

        # Cross Time Stage: Directly apply multi-head attention from temporal dimension
        time_attn_output, _, _ = self.time_attention(hidden_states)
        hidden_states = hidden_states + time_attn_output
        hidden_states = self.norm1(hidden_states)

        # FFN
        hidden_states = hidden_states + self.ffn1(hidden_states)
        hidden_states = self.norm2(hidden_states)

        # Cross Dimension Stage
        # use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        hidden_states = rearrange(hidden_states, "(b c) n d -> (b n) c d", b=batch_size)
        router_query = repeat(self.router, "n f d -> (b n) f d", b=batch_size)
        # cross attn: batch_router as query, dim_send as key, value
        buffer_states, _, _ = self.dim_sender(router_query, hidden_states)
        # cross attn: send_states as query, buffer_states as key, value
        feature_attn_output, _, _ = self.dim_receiver(hidden_states, buffer_states)
        hidden_states = hidden_states + feature_attn_output
        hidden_states = self.norm3(hidden_states)

        # FFN
        hidden_states = hidden_states + self.ffn2(hidden_states)
        hidden_states = self.norm4(hidden_states)
        hidden_states = rearrange(hidden_states, "(b n) c d -> (b c) n d", b=batch_size)

        return hidden_states


class CrossformerEncoder(nn.Module):
    """
    Crossformer Encoder
    """
    def __init__(self, encoder_layers: nn.ModuleList):
        super().__init__()
        self.encoder_layers = encoder_layers

    def forward(self, hidden_states: torch.Tensor) -> Iterable[torch.Tensor]:

        hidden_states_list = [hidden_states]
        for layer in self.encoder_layers:
            hidden_states_list.append(layer(hidden_states_list[-1]))
        return hidden_states_list


class CrossformerDecoderLayer(nn.Module):
    """
    Crossformer Decoder
    """
    def __init__(
            self,
            num_features: int,
            hidden_size: int,
            n_heads: int,
            intermediate_size: int,
            num_patches: int,
            factor: int = 10,
            hidden_act: str = "gelu",
            dropout: float =0.1):
        super().__init__()
        self.self_attn = CrossformerEncoderLayer(
            num_features=num_features,
            hidden_size=hidden_size,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
            num_patches=num_patches,
            factor=factor,
            hidden_act=hidden_act,
            dropout=dropout,
        )
        self.cross_attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = MLPLayer(
            hidden_size, intermediate_size, hidden_act=hidden_act, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:

        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size * num_features, num_patches, hidden_size]
            key_value_states (torch.Tensor): Input tensor of shape [batch_size * num_features, num_patches, hidden_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * num_features, num_patches, hidden_size]
        """

        # two-stage attention (actually is an encoder layer)
        hidden_states = self.self_attn(hidden_states)

        # cross attention
        cross_attn_output, _, _ = self.cross_attn(
            hidden_states, key_value_states
        )
        hidden_states = hidden_states + cross_attn_output
        hidden_states = self.cross_attn_norm(hidden_states)

        # FFN
        hidden_states = hidden_states + self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states)

        return hidden_states


class CrossformerDecoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, decoder_layers: nn.ModuleList):
        super().__init__()
        self.decoder_layers = decoder_layers

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_list: Optional[Iterable[torch.Tensor]] = None,
            ) -> Iterable[torch.Tensor]:

        hidden_states_list = []
        for i, layer in enumerate(self.decoder_layers):
            key_value_states = kv_list[i]
            hidden_states = layer(hidden_states,  key_value_states)
            hidden_states_list.append(hidden_states)
        return hidden_states_list
