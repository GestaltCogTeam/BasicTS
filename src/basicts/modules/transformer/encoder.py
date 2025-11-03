# pylint: disable=unused-argument

from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import torch
from torch import nn

from .utils import build_layer


class EncoderLayer(nn.Module):

    """
    BasicTS Transformer block.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, Any]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre"
    ):
        super().__init__()
        self.self_attn = self_attn
        self.ffn_layer = ffn_layer

        self.pre_attn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.pre_ffn_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.post_attn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)
        self.post_ffn_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)

    def self_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Self-attention sublayer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
            **kwargs: Additional keyword arguments for the self-attention layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output hidden states and attention weights.
        """

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        if not output_attentions:
            attn_weights = None

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        return hidden_states, attn_weights

    def ffn_forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:

        """
        Feed-forward network sublayer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            **kwargs: Additional keyword arguments for the feed-forward network.

        Returns:
            torch.Tensor: Output hidden states.
        """

        residual = hidden_states

        # Pre-LN
        if self.pre_ffn_norm is not None:
            hidden_states = self.pre_ffn_norm(hidden_states)

        # FFN
        ffn_output = self.ffn_layer(hidden_states, **kwargs)

        # Residual connection
        hidden_states = residual + ffn_output

        # Post-LN
        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Feed-forward of BasicTS Transformer encoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output hidden states and attention weights.
        """

        # Self-attention sublayer
        hidden_states, attn_weights = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # FFN sublayer
        hidden_states = self.ffn_forward(
            hidden_states=hidden_states
        )

        return hidden_states, attn_weights


class Encoder(nn.Module):

    """
    BasicTS Transformer encoder.
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = encoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:

        """
        Feed-forward of BasicTS Transformer encoder.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
            **kwargs: Additional keyword arguments for the encoder layers.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]]]: Output hidden states and attention weights.
        """

        attn_weights = []
        for layer in self.layers:
            hidden_states, attns = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs
            )
            if output_attentions:
                attn_weights.append(attns)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            attn_weights = None
        return hidden_states, attn_weights
