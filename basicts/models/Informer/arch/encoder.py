from typing import List, Optional, Tuple

import torch
from torch import nn


class InformerEncoder(nn.Module):

    """
    Informer encoder with additional convolutional layers.
    """

    def __init__(
        self,
        encoder_layers: nn.ModuleList,
        conv_layers: Optional[nn.ModuleList] = None,
        layer_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.conv_layers = conv_layers
        self.layer_norm = layer_norm
        self.num_encoder_layers = len(encoder_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:

        attn_weights = []

        for i in range(self.num_encoder_layers):
            hidden_states, attns = self.encoder_layers[i](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attn_weights.append(attns)
            if self.conv_layers is not None and i < self.num_encoder_layers - 1:
                hidden_states = self.conv_layers[i](hidden_states)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            attn_weights = None
        return hidden_states, attn_weights
