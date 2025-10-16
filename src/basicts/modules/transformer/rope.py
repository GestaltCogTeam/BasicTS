
from typing import Union

import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):

    """
    BasicTS Rotary Position Embedding (RoPE) module.

    This module implements Rotary Position Embedding as described in the paper
    Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
    link: https://arxiv.org/abs/2104.09864
    """

    def __init__(
            self,
            dim: int,
            max_position_embeddings: int = 2048,
            base: float = 10000,
            dtype: torch.dtype = torch.float32,
            device: Union[torch.device, None] = None):
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype).to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def apply_rope(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            position_ids: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = self._cos[position_ids].unsqueeze(1).to(dtype=q.dtype)
        sin = self._sin[position_ids].unsqueeze(1).to(dtype=q.dtype)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos", emb.cos().to(self.dtype), persistent=False)
        self.register_buffer("_sin", emb.sin().to(self.dtype), persistent=False)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            position_ids: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply and cache RoPE.

        Args:
            q (torch.Tensor): The query tensor with shape [batch_size, n_heads, seq_len, head_size].
            k (torch.Tensor): The key tensor with shape [batch_size, n_heads, seq_len, head_size].
            seq_len (int, optional): The sequence length. Defaults to None.
            position_ids (torch.Tensor): The position ids tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The query and key tensors with shape [batch_size, n_heads, seq_len, head_size].
        """

        max_pos = int(position_ids.max()) + 1
        if max_pos > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_pos, device=q.device)
        q, k = self.apply_rope(q, k, position_ids)
        return q, k
