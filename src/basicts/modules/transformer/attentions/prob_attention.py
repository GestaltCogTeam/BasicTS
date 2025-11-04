
import math
from typing import Optional, Tuple

import torch
from torch import nn


class ProbAttention(nn.Module):
    """
    Probabilistic Sparse Attention layer in Informer.
    Modified to follow BasicTS style with clear type hints and structure.
    """
    def __init__(self,
                 hidden_size: int,
                 n_heads: int,
                 factor: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.factor = factor

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Reshape input to (batch, heads, seq_len, head_dim)."""
        return x.view(x.size(0), seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _prob_qk(self,
                query: torch.Tensor,
                key: torch.Tensor,
                sample_k: int,
                n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Probabilistic sparse attention calculation."""
        batch_size, n_heads, k_len, _ = key.size()
        q_len = query.size(2)

        # Sample random keys for each query
        key_expand = key.unsqueeze(-3).expand(batch_size, n_heads, q_len, k_len, -1)
        index_sample = torch.randint(k_len, (q_len, sample_k))
        key_sample = key_expand[:, :, torch.arange(q_len).unsqueeze(1), index_sample, :]
        q_k_sample = torch.matmul(query.unsqueeze(-2), key_sample.transpose(-2, -1)).squeeze()

        # Select top-k queries based on sparsity measurement
        m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), k_len)
        m_top = m.topk(n_top, sorted=False)[1]

        # Calculate attention scores for selected queries
        query = query[torch.arange(batch_size)[:, None, None],
                         torch.arange(n_heads)[None, :, None], m_top, :]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        return scores, m_top

    def _get_context(self,
                     value: torch.Tensor,
                     scores: torch.Tensor,
                     index: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     output_attentions: bool = False
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Update context with selected attention scores."""

        batch_size, n_heads, _, q_len = scores.size()
        if attention_mask is None:
            context = value.mean(dim=-2, keepdim=True).expand(
                batch_size, n_heads, q_len, -1).clone()
        else:
            context = value.cumsum(dim=-2)
            # get prob mask
            attention_mask = attention_mask.expand(-1, n_heads, -1, -1)
            batch_idx = torch.arange(batch_size, device=attention_mask.device)[:, None, None]
            head_idx = torch.arange(n_heads, device=attention_mask.device)[None, :, None]
            attention_mask = attention_mask[batch_idx, head_idx, index, :]
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context[torch.arange(batch_size)[:, None, None],
               torch.arange(n_heads)[None, :, None], index, :] = torch.matmul(attn, value)

        if output_attentions:
            attn_weights = torch.ones(
                batch_size, n_heads, q_len, q_len, device=value.device) / q_len
            attn_weights[torch.arange(batch_size)[:, None, None],
                     torch.arange(n_heads)[None, :, None],
                     index, :] = attn
        else:
            attn_weights = None
        return context, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross = key_value_states is not None
        kv_seq_len = key_value_states.size(1) if is_cross else hidden_states.size(1)
        batch_size, q_seq_len, _ = hidden_states.size()

        # Project inputs
        query = self._shape(self.q_proj(hidden_states), q_seq_len)
        key = self._shape(self.k_proj(key_value_states if is_cross else hidden_states), kv_seq_len)
        value = self._shape(self.v_proj(key_value_states if is_cross else hidden_states), kv_seq_len)

        # Calculate sparse attention
        sample_k = min(self.factor * math.ceil(math.log(kv_seq_len)), kv_seq_len)
        n_top = min(self.factor * math.ceil(math.log(q_seq_len)), q_seq_len)

        scores, index = self._prob_qk(query, key, sample_k, n_top)

        # Calculate context
        context, attn_weights = self._get_context(
            value, scores, index, attention_mask, output_attentions)

        # Reshape output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.o_proj(context)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights, None
