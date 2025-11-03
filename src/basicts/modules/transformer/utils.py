from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn


def prepare_causal_attention_mask(
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None, # padding mask
    past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Generate a causal attention mask (including padding mask) for the decoder.
    
    Args:
        input_shape: (batch_size, seq_len)
        inputs_embeds: Input embeddings tensor, used to determine device and dtype.
        attention_mask: (batch_size, seq_len), optional. Padding mask. 1=valid, 0=padding.
        past_key_values_length: Length of past key values cache (for incremental decoding).
    
    Returns:
        combined_attention_mask: (batch_size, 1, tgt_len, tgt_len + past_key_values_length)
    """
    batch_size, seq_len = input_shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # 1. 创建因果掩码（下三角矩阵）
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1
    ) # [seq_len, seq_len]
    causal_mask = ~causal_mask

    if past_key_values_length > 0:
        causal_mask = torch.cat([
            torch.ones((seq_len, past_key_values_length), device=device, dtype=torch.bool),
            causal_mask
        ], dim=-1)  # shape: [seq_len, past_key_values_length + seq_len]

    # # [batch_size, 1, seq_len, seq_len + past_key_values_length]
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if attention_mask is not None:
        assert attention_mask.dim() == 2, \
            "attention_mask should be in shape [batch_size, seq_len]"
        padding_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        combined_mask = causal_mask & padding_mask
    else:
        combined_mask = causal_mask

    combined_mask = combined_mask.to(dtype)
    combined_mask = (1.0 - combined_mask) * torch.finfo(dtype).min
    return combined_mask

def build_layer(
    layer: Union[Callable, Tuple[Callable, Any]],
) -> nn.Module:
    """Build layer.

    Args:
        layer (Callable | Tuple[Callable, Any]): Layer norm function or tuple of layer norm function and kwargs.

    Returns:
        nn.Module: Layer module.
    """
    if isinstance(layer, Callable):
        return layer()
    if isinstance(layer, tuple):
        norm_fn, norm_args = layer
        if isinstance(norm_args, dict):
            return norm_fn(**norm_args)
        elif isinstance(norm_args, (list, tuple)):
            return norm_fn(*norm_args)
        else:
            return norm_fn(norm_args)
    else:
        raise ValueError(f"layer_norm should be Callable or Tuple[Callable, Any], but got {type(layer)}.")
