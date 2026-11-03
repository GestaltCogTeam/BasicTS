# cats_layers.py

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from basicts.modules.transformer import DecoderOnlyLayer, MultiHeadAttention


class GEGLU(nn.Module):
    """
    Gated GELU feed-forward activation used in CATS FFN blocks.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (x, gate) and apply GELU to the gate.

        Args:
            x: [..., 2 * D]

        Returns:
            [..., D]
        """
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class QueryAdaptiveMasking(nn.Module):
    """
    Query-Adaptive Masking (QAM) for CATS.

    It drops tokens along a specific dimension (typically patch index) with a
    probability that linearly increases from `start_prob` to `end_prob`.

    This module is only active during training.
    """

    def __init__(self, dim=1, start_prob=0.1, end_prob=0.5) -> None:
        """
        Args:
            dim: dimension along which dropout schedule is defined.
            start_prob: initial dropout probability at index 0.
            end_prob: final dropout probability at the last index.
        """
        super().__init__()
        self.dim = dim
        self.start_prob = start_prob
        self.end_prob = end_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            Masked tensor during training; original tensor during evaluation.
        """
        if not self.training:
            return x

        size = x.shape[self.dim]
        # Linearly interpolated dropout rate along the target dimension.
        dropout_prob = torch.linspace(
            self.start_prob,
            self.end_prob,
            steps=size,
            device=x.device,
            dtype=x.dtype,
        )
        view_shape = [-1 if i == self.dim else 1 for i in range(x.dim())]
        dropout_prob = dropout_prob.view(view_shape)
        mask = torch.bernoulli(1.0 - dropout_prob)
        return x * mask


class CATSDecoderLayer(DecoderOnlyLayer):
    """
    One CATS decoder layer operating on patch-level sequences.

    This layer:
        - Uses CATSAttention (based on BasicTS MultiHeadAttention) as a
          cross-attention between history patches (encoder memory) and
          prediction patches (decoder queries).
        - Applies Query-Adaptive Masking (QAM) to both attention output and
          FFN output.
        - Uses a GEGLU-based FFN block as in the original CATS implementation.
        - Supports residual attention scores across layers.

    Constructor style:
        CATSPatchDecoderLayer(config)
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: configuration object with at least:
                - hidden_size
                - n_heads
                - intermediate_size
                - dropout
                - qam_start
                - qam_end
                - res_attention (bool, optional)
                - store_attn (bool, optional)
        """
        # ---- Attention ----
        cross_attn = MultiHeadAttention(
            config.hidden_size, config.n_heads, config.dropout
        )

        # ---- FFN with GEGLU ----
        ffn_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=True),
            GEGLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=True),
        )

        # Now register inside super().__init__
        super().__init__(
            self_attn=cross_attn, ffn_layer=ffn_layer, layer_norm=lambda: None
        )

        self.qam_start = config.qam_start
        self.qam_end = config.qam_end

        # LayerNorm configuration for DecoderOnlyLayer / EncoderLayer.
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # QAM is applied along the patch dimension (dim = 1).
        self.dropout_attn = QueryAdaptiveMasking(
            dim=1, start_prob=self.qam_start, end_prob=self.qam_end
        )
        self.dropout_ffn = QueryAdaptiveMasking(
            dim=1,
            start_prob=self.qam_start,
            end_prob=self.qam_end,
        )

    def forward(
        self,
        seq_patches: Tensor,
        pred_patches: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass of one CATS patch decoder layer.

        Args:
            seq_patches: history patches, shape [B * N, S, D],
                used as encoder memory (keys/values).
            pred_patches: prediction patches, shape [B * N, P, D],
                used as decoder queries.
            prev_scores: previous attention logits for residual attention,
                shape [B, n_heads, P, S] or None.
            attention_mask: attention mask broadcastable to
                [B, n_heads, P, S].
            output_attentions: whether to return attention weights.

        Returns:
            seq_patches: unchanged history patches.
            pred_patches: updated prediction patches, shape [B * N, P, D].
            scores: current attention logits if `self.res_attention` is True,
                otherwise None.
        """
        # -------- Attention sub-layer --------
        residual = pred_patches

        attn_output, attn_weights, _ = self.self_attn(
            hidden_states=pred_patches,
            key_value_states=seq_patches,
        )

        # Residual connection + QAM
        pred_patches = residual + self.dropout_attn(attn_output)

        pred_patches = self.layer_norm(pred_patches)

        # -------- FFN sub-layer --------
        residual = pred_patches
        ffn_output = self.ffn_layer(pred_patches)
        pred_patches = residual + self.dropout_ffn(ffn_output)

        pred_patches = self.layer_norm(pred_patches)

        return seq_patches, pred_patches


class CATSDecoder(nn.Module):
    """
    A stack of CATS patch decoder layers.

    This module preserves the original CATS behavior:
        - If `config.res_attention=True`, attention scores are accumulated across
          layers.
        - Only the prediction patches are updated and returned.

    Constructor style:
        CATSPatchDecoder(config)
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: configuration object with at least:
                - hidden_size
                - n_heads
                - intermediate_size
                - dropout
                - qam_start
                - qam_end
                - store_attn
                - num_layers
        """
        super().__init__()
        # Number of layers: prefer num_layers (as in many TS models),
        # fallback to e_layers if needed.
        self.n_layers = config.num_layers

        self.layers = nn.ModuleList(
            [CATSDecoderLayer(config) for _ in range(self.n_layers)]
        )

    def forward(
        self,
        seq_patches: Tensor,
        pred_patches: Tensor,
    ) -> Tensor:
        """
        Args:
            seq_patches: history patches, shape [B * N, S, D].
            pred_patches: prediction patches, shape [B * N, P, D].
            attention_mask: attention mask broadcastable to
                [B, n_heads, P, S].
            output_attentions: whether underlying layers store attention maps.

        Returns:
            Updated prediction patches, shape [B * N, P, D].
        """
        # scores is ignored because it is None
        for layer in self.layers:
            seq_patches, pred_patches = layer(
                seq_patches=seq_patches,
                pred_patches=pred_patches,
            )
        return pred_patches
