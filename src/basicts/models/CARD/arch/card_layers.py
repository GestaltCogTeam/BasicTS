import torch
from torch import nn
from .card_attention import CARDAttenion, Transpose

from basicts.modules.transformer import EncoderLayer, Encoder


class CARDEncoderLayer(EncoderLayer):
    """
    One encoder layer of the CARD model.

    Each layer consists of:
        - Channel attention (cross-channel interaction)
        - Token attention (temporal interaction)
        - Feed-forward MLP and Dropout
        - Batch normalization and residual connection

    This design follows the structure of AutoformerEncoderLayer
    for better modularity and readability.
    """

    def __init__(self, config) -> None:
        # Initialize parent with dummy self_attn/ffn; we override forward anyway.
        super().__init__(
            self_attn=nn.Identity(),
            ffn_layer=nn.Identity(),
            layer_norm=(nn.LayerNorm, config.hidden_size),
            norm_position="pre",
        )

        # Channel-wise attention (over hidden dimension)
        self.channel_attn = CARDAttenion(config, over_hidden=True)
        # Token-wise attention
        self.token_attn = CARDAttenion(config, over_hidden=False)

        # Simple linear projection + dropout
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Temporal BatchNorm with transpose wrapper
        self.norm = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.hidden_size, momentum=config.momentum),
            Transpose(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """
        Forward pass of a single CARD encoder layer.

        Args:
            hidden_states: [B, C, T, D]
                B - batch size
                C - number of variables
                T - temporal length
                D - feature dimension

        Returns:
            (Tensor, None):
                - updated hidden states [B, C, T, D]
                - attention weights (not used, so None)
        """
        x = hidden_states  # [B, C, T, D]

        # (1) Channel-wise attention: operate across channels
        # Permute to [B, T, C, D] for over_hidden mode, then back.
        out_channel = self.channel_attn(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # (2) Token-wise attention: operate across temporal tokens
        out_token = self.token_attn(out_channel)

        # (3) Residual + projection + dropout
        out = self.dropout(self.mlp(out_channel + out_token)) + x

        # (4) Temporal BatchNorm per variable
        b, c, t, d = out.shape
        out = self.norm(out.reshape(b * c, t, d)).reshape(b, c, t, d)

        # CARD does not expose attention maps at layer level
        attn_weights = None
        return out, attn_weights


class CARDEncoder(Encoder):
    """
    Multi-layer encoder of CARD.

    Stacks multiple CARDEncoderLayers and reuses BasicTS Encoder interface.
    """

    def __init__(self, config) -> None:
        layers = nn.ModuleList(
            [CARDEncoderLayer(config) for _ in range(config.num_layers)]
        )
        # Use parent Encoder to manage layer stack
        super().__init__(encoder_layers=layers, layer_norm=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through all CARD encoder layers.

        Args:
            hidden_states: [B, C, T, D]

        Returns:
            Tensor: encoded representation [B, C, T, D]
        """
        hidden_states, _ = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        # Only return encoded states to keep the original CARD interface
        return hidden_states
