from typing import Optional, Tuple

import torch
from torch import nn

from ..kv_cache import KVCache
from ..rope import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    """
    BasicTS Multi-Head Attention module.
    
    Features:
        - Can be used as self-/cross-attention.
        - MHA/MQA/GQA with various Key-Value heads.
        - Support KV cache.
        - Support RoPE.

    """
    def __init__(self,
                 hidden_size: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 kv_heads: Optional[int] = None,
                 rope: Optional[RotaryPositionEmbedding] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.kv_heads = kv_heads or n_heads
        self.num_kv_groups = self.n_heads // self.kv_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_heads * self.head_size)
        self.v_proj = nn.Linear(hidden_size, self.kv_heads * self.head_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def _shape_q(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(x.size(0), seq_len, self.n_heads, self.head_size).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(x.size(0), seq_len, self.kv_heads, self.head_size).transpose(1, 2)

    def _repeat_kv(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repeat key/value for GQA.
        """
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:

        # Query
        B, L, _ = hidden_states.size()
        query = self._shape_q(self.q_proj(hidden_states), L)

        is_cross = key_value_states is not None

        # Key/Value
        if is_cross: # cross-attn (typically does not use rope)
            if use_cache:
                # first time, cache key/value
                if len(past_key_value) <= layer_idx:
                    kv_seq_len = key_value_states.size(1)
                    key = self._shape_kv(self.k_proj(key_value_states), kv_seq_len)
                    value = self._shape_kv(self.v_proj(key_value_states), kv_seq_len)
                    past_key_value.update(key, value, layer_idx)
                else: # from kv cache
                    key, value = past_key_value[layer_idx]
            else:
                kv_seq_len = key_value_states.size(1)
                key = self._shape_kv(self.k_proj(key_value_states), kv_seq_len)
                value = self._shape_kv(self.v_proj(key_value_states), kv_seq_len)
        else: # self-attn
            # compute key/value from hidden_states
            key = self._shape_kv(self.k_proj(hidden_states), L)
            value = self._shape_kv(self.v_proj(hidden_states), L)
            # rope
            if self.rope is not None:
                if position_ids is None:
                    raise ValueError("`position_ids` is required when `rope` is not None.")
                query, key = self.rope(query, key, position_ids)
            if use_cache:
                key, value = past_key_value.update(key, value, layer_idx)

        # repeat kv for GQA
        key, value = self._repeat_kv(key, value)

        # attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        output = self.out_proj(context)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights, past_key_value
