# pylint: disable=not-callable
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..kv_cache import KVCache


class AutoCorrelation(nn.Module):
    """
    Auto correlation layer from Autoformer.
    """
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, factor: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj   = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj   = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.factor = factor

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(x.size(0), seq_len, self.n_heads, self.head_dim)

    def _time_delay_aggregation(self, x: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        :param x:    [batch_size, n_heads, head_dim, seq_len]
        :param corr: [batch_size, n_heads, head_dim, seq_len]
        :return: [batch_size, n_heads, head_dim, seq_len]
        """
        B, _, _, seq_len = x.shape
        top_k = int(self.factor * math.log(seq_len))
        mean_corr = corr.mean(dim=(1, 2))

        agg = torch.zeros_like(x)
        if self.training:
            _, top_idx = torch.topk(mean_corr.mean(dim=0), top_k, dim=-1)
            weights = torch.stack([mean_corr[:, idx] for idx in top_idx], dim=-1)
            weights = F.softmax(weights, dim=-1)
            for i, idx in enumerate(top_idx):
                rolled = torch.roll(x, shifts=-int(idx), dims=-1)
                agg += rolled * weights[:, i, None, None, None]
            return agg
        else:
            weights, delays = torch.topk(mean_corr, top_k, dim=-1)  # [B, top_k]
            weights = F.softmax(weights, dim=-1)
            base_idx = torch.arange(seq_len, device=x.device)[None, None, None, :].expand_as(x)
            x_ext = x.repeat(1, 1, 1, 2)  # for wrapping
            for i in range(top_k):
                delay_idx = base_idx + delays[:, i, None, None, None]
                pattern = torch.gather(x_ext, dim=-1, index=delay_idx.expand(B, self.n_heads, self.head_dim, seq_len))
                agg += pattern * weights[:, i, None, None, None]
            return agg

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[KVCache] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            layer_idx: Optional[int] = None,
            )->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # Query
        B, q_len, _ = hidden_states.size()
        query = self._shape(self.q_proj(hidden_states), q_len)

        is_cross = key_value_states is not None

        # Key/Value
        if is_cross:
            if use_cache:
                # first time, cache key/value
                if len(past_key_value) <= layer_idx:
                    kv_len = key_value_states.size(1)
                    key = self._shape(self.k_proj(key_value_states), kv_len)
                    value = self._shape(self.v_proj(key_value_states), kv_len)
                    past_key_value.update(key, value, layer_idx)
                else: # from kv cache
                    key, value = past_key_value[layer_idx]
            else:
                kv_len = key_value_states.size(1)
                key = self._shape(self.k_proj(key_value_states), kv_len)
                value = self._shape(self.v_proj(key_value_states), kv_len)
        else: # self-attn
            # compute key/value from hidden_states
            key = self._shape(self.k_proj(hidden_states), q_len)
            value = self._shape(self.v_proj(hidden_states), q_len)
            if use_cache:
                key, value = past_key_value.update(key, value, layer_idx)

        # Align size to [batch_size, q_len, n_heads, head_dim]
        kv_len = value.size(1)
        if q_len > kv_len:
            pad_shape = (0, 0, 0, 0, 0, q_len - kv_len)
            key = F.pad(key, pad_shape, mode="constant", value=0.0)
            value = F.pad(value, pad_shape, mode="constant", value=0.0)
        else:
            value, key = value[:, :q_len], key[:, :q_len]

        q_fft = torch.fft.rfft(query.movedim(1, -1), dim=-1)
        k_fft = torch.fft.rfft(key.movedim(1, -1), dim=-1)
        corr  = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)

        # attention mask is not actually used in Autoformer
        if attention_mask is not None:
            corr = corr + attention_mask

        # [batch_size, n_heads, head_dim, seq_len]
        context = self._time_delay_aggregation(value.movedim(1, -1), corr)
        context = context.movedim(-1, 1).contiguous()   # [batch_size, q_len, n_heads, head_dim]

        out = self.o_proj(context.view(B, q_len, -1))

        corr = corr.movedim(-1, 1) if output_attentions else None

        return out, corr, past_key_value
