from typing import Optional, Tuple

import torch
from torch import nn

from basicts.modules.transformer import EncoderLayer, Seq2SeqDecoderLayer


class DSAttention(nn.Module):

    """
    De-stationary Attention Layer.
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(x.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Forward pass of the De-stationary Attention Layer.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            key_value_states (Optional[torch.Tensor]): Optional tensor of shape [batch_size, seq_len, hidden_size] for cross-attention.
            attention_mask (Optional[torch.Tensor]): Optional tensor of shape [batch_size, 1, 1, seq_len] for masking.
            output_attentions (bool): Whether to return attention weights.
            tau (Optional[torch.Tensor]): Optional tensor of shape [batch_size, 1, 1, 1] for de-stationary attention.
            delta (Optional[torch.Tensor]): Optional tensor of shape [batch_size, 1, 1, seq_len] for de-stationary attention.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor of shape [batch_size, seq_len, hidden_size] and optional attention weights.
        """

        # Query
        B, L, _ = hidden_states.size()
        query = self._shape(self.q_proj(hidden_states), L)

        is_cross = key_value_states is not None

        # Key/Value
        if is_cross: # cross-attn (typically does not use rope)
            kv_len = key_value_states.size(1)
            key = self._shape(self.k_proj(key_value_states), kv_len)
            value = self._shape(self.v_proj(key_value_states), kv_len)
        else: # self-attn
            # compute key/value from hidden_states
            key = self._shape(self.k_proj(hidden_states), L)
            value = self._shape(self.v_proj(hidden_states), L)

        # tau and delta for de-stationary attention
        # tau -> [batch_size, 1, 1, 1], delta -> [batch_size, 1, 1, seq_len]
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        # attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) * tau + delta
        scores /= (self.head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        output = self.out_proj(context)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights, None


class NonstationaryTransformerEncoderLayer(EncoderLayer):

    """
        Encoder layer for non-stationary transformer.
        Extra arguments `tau` and `delta` are passed for de-stationary attention.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Forward pass of the NonstationaryTransformerEncoderLayer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            tau (Optional[torch.Tensor]):
                The tau for de-stationary attention. Defaults to None.
            delta (Optional[torch.Tensor]):
                The delta for de-stationary attention. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                The output hidden states and optional attentions.
        """

        # Self-attention sublayer
        hidden_states, attn_weights = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            tau=tau,
            delta=delta
        )

        # FFN sublayer
        hidden_states = self.ffn_forward(
            hidden_states=hidden_states
        )

        return hidden_states, attn_weights


class NonstationaryTransformerDecoderLayer(Seq2SeqDecoderLayer):

    """
        Decoder layer for non-stationary transformer.
        Extra arguments `tau` and `delta` are passed for de-stationary attention.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        """
        Forward pass of the NonstationaryTransformerDecoderLayer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor]):
                The encoder attention mask. Defaults to None.
            tau (Optional[torch.Tensor]):
                The tau for de-stationary attention. Defaults to None.
            delta (Optional[torch.Tensor]):
                The delta for de-stationary attention. Defaults to None.
        """

        # Self-attention sublayer
        hidden_states, self_attn_weights = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            tau=tau
        )

        # Cross-attention sublayer
        if self.cross_attn is not None and key_value_states is not None:
            hidden_states, cross_attn_weights = self.cross_attn_forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                tau=tau,
                delta=delta
            )
        else:
            cross_attn_weights = None

        # Feed-forward sublayer
        hidden_states = self.ffn_forward(hidden_states)

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        return hidden_states, self_attn_weights, cross_attn_weights

class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(
            self,
            num_features: int,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            kernel_size: int = 3):
        super().__init__()

        self.series_conv = nn.Conv1d(
            input_size, 1, kernel_size, padding=1, padding_mode="circular", bias=False)

        layers = [nn.Linear(2 * num_features, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size, bias=False)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.series_conv(x)
        # [batch_size, 2 * hidden_size]
        x = torch.cat([x, stats], dim=1).view(batch_size, -1)
        return self.mlp(x)
