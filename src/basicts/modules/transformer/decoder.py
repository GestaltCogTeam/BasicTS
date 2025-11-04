# pylint: disable=unused-argument

from typing import Callable, Literal, Optional, Tuple, Union

import torch
from torch import nn

from .encoder import EncoderLayer
from .kv_cache import KVCache
from .utils import build_layer


class DecoderOnlyLayer(EncoderLayer):

    """
    BasicTS Transformer decoder layer for decoder-only architecture.
    It is a variant of the encoder layer, with improvements for autoregressive decoding.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, dict]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre",
    ):

        """
        Args:
            self_attn (nn.Module):
                The self-attention layer.
            ffn_layer (nn.Module):
                The feed-forward network layer.
            layer_norm (Union[Callable, Tuple[Callable, dict]]):
                The layer normalization layer. If a tuple is provided, the first element is 
                the layer normalization class, and the second element is the kwargs for
                  the layer normalization.
            norm_position (Literal["pre", "post", "sandwich"], optional):
                The position of the layer normalization layer. Defaults to "pre".
        """
        super().__init__(self_attn, ffn_layer, layer_norm, norm_position)

    def self_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs
    ):

        """
        Forward pass of the self-attention layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                The position ids. Defaults to None.
            past_key_value (Optional[KVCache], optional):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments for the self-attention layer.
        """

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
            **kwargs
        )

        if not output_attentions:
            attn_weights = None
        if not use_cache:
            present_key_value = None

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        return hidden_states, attn_weights, present_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        """
        Forward pass of the decoder layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                The position ids. Defaults to None.
            past_key_value (Optional[KVCache], optional):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

        # Self-attention sublayer
        hidden_states, attn_weights, present_key_value = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
        )

        # FFN sublayer
        hidden_states = self.ffn_forward(hidden_states)

        return hidden_states, attn_weights, present_key_value


class AutoRegressiveDecoder(nn.Module):

    """
    Auto-regressive decoder for decoder-only architecture.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        """
        Args:
            decoder_layers (nn.ModuleList):
                The decoder layers.
            layer_norm (Optional[nn.Module], optional):
                The layer norm. Defaults to None.
        """

        super().__init__()
        self.layers = decoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[KVCache] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[KVCache]]:
        """
        Forward pass of the auto-regressive decoder.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                The position ids. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            kv_cache (Optional[KVCache], optional):
                The key value cache. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

        if use_cache and kv_cache is None:
            kv_cache = KVCache()
        seq_len = hidden_states.size(1)
        past_kv_len = 0
        if use_cache:
            past_kv_len = kv_cache.get_seq_length()

        if position_ids is None:
            position_ids = torch.arange(
                past_kv_len, seq_len + past_kv_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        attn_weights = ()
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, attns, kv_cache = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=kv_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
                **kwargs
            )
            if output_attentions:
                attn_weights += (attns,)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            attn_weights = None
        return hidden_states, attn_weights, kv_cache


class Seq2SeqDecoderLayer(EncoderLayer):

    """
    BasicTS Transformer decoder layer for encoder-decoder architecture.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, dict]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre",
    ):
        super().__init__(self_attn, ffn_layer, layer_norm, norm_position)
        self.cross_attn = cross_attn
        self.pre_cross_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.post_cross_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)

    def self_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Forward pass of the self-attention layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                The output hidden states and attention weights.
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

    def cross_attn_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the cross-attention layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            past_key_value (Optional[Tuple[KVCache]]):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments for cross-attention.
        """
        residual = hidden_states
        if self.pre_cross_norm is not None:
            hidden_states = self.pre_cross_norm(hidden_states)
        cross_output, cross_attn_weights, _ = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = residual + cross_output
        if self.post_cross_norm is not None:
            hidden_states = self.post_cross_norm(hidden_states)
        return hidden_states, cross_attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        """
        Forward pass of the Seq2SeqDecoderLayer.

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
            **kwargs: Additional keyword arguments.
        """

        # Self-attention sublayer
        hidden_states, self_attn_weights = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # Cross-attention sublayer
        if self.cross_attn is not None and key_value_states is not None:
            hidden_states, cross_attn_weights = self.cross_attn_forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
        else:
            cross_attn_weights = None

        # Feed-forward sublayer
        hidden_states = self.ffn_forward(hidden_states)

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        return hidden_states, self_attn_weights, cross_attn_weights


class Seq2SeqDecoderLayerV2(nn.Module):

    """
    Seq2SeqDecoderLayerV2 is specially designed for time series foundation models.
    It adds kv cache for self and cross-attention to speed up inference.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        ffn_layer: nn.Module,
        layer_norm: Union[Callable, Tuple[Callable, dict]],
        norm_position: Literal["pre", "post", "sandwich"] = "pre",
    ):
        super().__init__(self_attn, ffn_layer, layer_norm, norm_position)
        self.cross_attn = cross_attn
        self.pre_cross_norm = None if norm_position == "post" \
            else build_layer(layer_norm)
        self.post_cross_norm = None if norm_position == "pre" \
            else build_layer(layer_norm)

    def self_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs
    ):

        """
        Forward pass of the self-attention layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask. Defaults to None.
            past_key_value (Optional[KVCache], optional):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                The output hidden states, attention weights, and present key value cache.
        """

        residual = hidden_states

        # Pre-LN
        if self.pre_attn_norm is not None:
            hidden_states = self.pre_attn_norm(hidden_states)

        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
            **kwargs
        )

        if not output_attentions:
            attn_weights = None
        if not use_cache:
            present_key_value = None

        # Residual connection
        hidden_states = residual + attn_output

        # Post-LN
        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        return hidden_states, attn_weights, present_key_value

    def cross_attn_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[KVCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[KVCache]]]:
        """
        Forward pass of the cross-attention layer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            past_key_value (Optional[Tuple[KVCache]]):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments for cross-attention.
        """
        residual = hidden_states
        if self.pre_cross_norm is not None:
            hidden_states = self.pre_cross_norm(hidden_states)
        cross_output, cross_attn_weights, cross_present_kv = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
            **kwargs
        )
        hidden_states = residual + cross_output
        if self.post_cross_norm is not None:
            hidden_states = self.post_cross_norm(hidden_states)
        return hidden_states, cross_attn_weights, cross_present_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[KVCache]] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[KVCache]]]:

        """
        Forward pass of the Seq2SeqDecoderLayer.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor]):
                The encoder attention mask. Defaults to None.
            past_key_value (Optional[Tuple[KVCache]]):
                The past key value cache. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            layer_idx (Optional[int], optional):
                The layer index. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

        self_past_kv, cross_past_kv = past_key_value[0], past_key_value[1] \
            if past_key_value is not None else (None, None)

        # Self-attention sublayer
        hidden_states, self_attn_weights, self_present_kv = self.self_attn_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_past_kv,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
        )

        # Cross-attention sublayer
        if self.cross_attn is not None and key_value_states is not None:
            hidden_states, cross_attn_weights, cross_present_kv = self.cross_attn_forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
            )
        else:
            cross_attn_weights = None
            cross_present_kv = None

        hidden_states = self.ffn_forward(hidden_states)

        if not output_attentions:
            self_attn_weights = cross_attn_weights = None

        present_key_value = (self_present_kv, cross_present_kv) if use_cache else None

        return hidden_states, self_attn_weights, cross_attn_weights, present_key_value


class Seq2SeqDecoder(nn.Module):
    """
    Seq2Seq decoder for encoder-decoder architecture.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        """
        Args:
            decoder_layers (nn.ModuleList):
                The decoder layers.
            layer_norm (Optional[nn.Module], optional):
                The layer norm. Defaults to None.
        """

        super().__init__()
        self.layers = decoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor,
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[KVCache]]]:
        """
        Forward pass of the Seq2SeqDecoder.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor]):
                The encoder attention mask. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            kv_cache (Optional[Tuple[KVCache]], optional):
                The key value cache. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

        self_attn_weights, cross_attn_weights = (), ()
        for layer in self.layers:
            hidden_states, self_attns, cross_attns = layer(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs
            )
            if output_attentions:
                self_attn_weights += (self_attns,)
                cross_attn_weights += (cross_attns,)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            self_attn_weights = cross_attn_weights = None
        return hidden_states, self_attn_weights, cross_attn_weights


class Seq2SeqDecoderV2(nn.Module):
    """
    Seq2Seq decoder for encoder-decoder architecture.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        layer_norm: Optional[nn.Module] = None,
    ):
        """
        Args:
            decoder_layers (nn.ModuleList):
                The decoder layers.
            layer_norm (Optional[nn.Module], optional):
                The layer norm. Defaults to None.
        """

        super().__init__()
        self.layers = decoder_layers
        self.layer_norm = layer_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[Tuple[KVCache]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor,
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[torch.Tensor]],
               Optional[Tuple[KVCache]]]:
        """
        Forward pass of the Seq2SeqDecoder.

        Args:
            hidden_states (torch.Tensor):
                The input hidden states.
            key_value_states (Optional[torch.Tensor]):
                The key value states for cross-attention. Defaults to None.
            attention_mask (Optional[torch.Tensor]):
                The attention mask. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor]):
                The encoder attention mask. Defaults to None.
            output_attentions (bool, optional):
                Whether to output attentions. Defaults to False.
            use_cache (bool, optional):
                Whether to use cache. Defaults to False.
            kv_cache (Optional[Tuple[KVCache]], optional):
                The key value cache. Defaults to None.
            **kwargs: Additional keyword arguments.
        """

        if use_cache and kv_cache is None:
            kv_cache = (KVCache(), KVCache())

        self_attn_weights, cross_attn_weights = (), ()
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, self_attns, cross_attns, kv_cache = layer(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=kv_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=layer_idx,
                **kwargs
            )
            if output_attentions:
                self_attn_weights += (self_attns,)
                cross_attn_weights += (cross_attns,)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if not output_attentions:
            self_attn_weights = cross_attn_weights = None
        if not use_cache:
            kv_cache = None
        return hidden_states, self_attn_weights, cross_attn_weights, kv_cache
