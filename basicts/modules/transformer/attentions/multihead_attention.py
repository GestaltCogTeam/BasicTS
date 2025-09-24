from typing import Optional, Tuple

import torch
from torch import nn

from ..kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    """
    BasicTS Multi-Head Attention module.
    """
    def __init__(self,
                 hidden_size: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 use_rope: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            pass # TODO: support Rope

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(x.size(0), seq_len, self.n_heads, self.head_dim).transpose(1, 2)

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
        query = self._shape(self.q_proj(hidden_states), L)

        is_cross = key_value_states is not None

        # Key/Value
        if is_cross:
            if use_cache:
                # first time, cache key/value
                if len(past_key_value) <= layer_idx:
                    kv_seq_len = key_value_states.size(1)
                    key = self._shape(self.k_proj(key_value_states), kv_seq_len)
                    value = self._shape(self.v_proj(key_value_states), kv_seq_len)
                    past_key_value.update(key, value, layer_idx)
                else: # from kv cache
                    key, value = past_key_value[layer_idx]
            else:
                kv_seq_len = key_value_states.size(1)
                key = self._shape(self.k_proj(key_value_states), kv_seq_len)
                value = self._shape(self.v_proj(key_value_states), kv_seq_len)
        else: # self-attn
            # compute key/value from hidden_states
            key = self._shape(self.k_proj(hidden_states), L)
            value = self._shape(self.v_proj(hidden_states), L)
            if use_cache:
                key, value = past_key_value.update(key, value, layer_idx)

        # rope
        if self.use_rope:
            if position_ids is None:
                raise ValueError("`position_ids` is required when `use_rope` is True.")
            pass # TODO: support Rope

        # attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
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
