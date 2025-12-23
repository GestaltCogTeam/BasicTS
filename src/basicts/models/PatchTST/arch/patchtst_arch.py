# Cell
from typing import List, Optional, Tuple

import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.embed import PatchEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import (Encoder, EncoderLayer,
                                         MultiHeadAttention)

from ..config.patchtst_config import PatchTSTConfig
from .patchtst_layers import PatchTSTBatchNorm, PatchTSTHead


class PatchTSTBackbone(nn.Module):
    """
    Paper: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    Link: https://arxiv.org/abs/2211.14730
    Official Code: https://github.com/yuqinie98/PatchTST
    Venue: ICLR 2023
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: PatchTSTConfig):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()

        self.num_features = config.num_features

        # patching and embedding
        padding = (0, config.patch_stride) if config.padding else None
        self.patch_embedding = PatchEmbedding(
            config.hidden_size, config.patch_len, config.patch_stride, padding, config.fc_dropout)
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1)
        if config.padding:
            self.num_patches += 1

        # Encoder
        norm_type = nn.LayerNorm if config.norm_type == "layer_norm" else PatchTSTBatchNorm
        self.encoder = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    MultiHeadAttention(config.hidden_size, config.n_heads, config.attn_dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.fc_dropout),
                    layer_norm=(norm_type, config.hidden_size),
                    norm_position="post"
                ) for _ in range(config.num_layers)
            ])
        )
        self.output_attentions = config.output_attentions

    def forward(
            self, inputs: torch.Tensor
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, num_features, num_patches, hidden_size]
            Optional[List[torch.Tensor]]: attention weights if output_attentions=True, else None
        """

        hidden_states = self.patch_embedding(inputs)
        hidden_states, attn_weights= self.encoder(hidden_states, output_attentions=self.output_attentions)
        hidden_states = hidden_states.reshape(
            -1, self.num_features, hidden_states.shape[-2], hidden_states.shape[-1])
        return hidden_states, attn_weights


class PatchTSTForForecasting(nn.Module):
    """
    PatchTST for time series forecasting.
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.decomp = config.decomp
        if self.decomp:
            self.decomp_layer = MovingAverageDecomposition(config.moving_avg)
            self.seasonal_backbone = PatchTSTBackbone(config)
            self.trend_backbone = PatchTSTBackbone(config)
            self.num_patches = self.seasonal_backbone.num_patches
        else:
            self.backbone = PatchTSTBackbone(config)
            self.num_patches = self.backbone.num_patches
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecasting_head = PatchTSTHead(
            self.num_patches * config.hidden_size,
            config.output_len,
            config.individual_head,
            config.num_features,
            config.head_dropout)
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last)
        self.output_attentions = config.output_attentions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of PatchTSTForForecasting.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        # [batch_size, num_features, num_patches, hidden_size]
        if self.decomp:
            seasonal_hidden_states, attn_weights = self.seasonal_backbone(inputs)
            trend_hidden_states, _ = self.trend_backbone(inputs)
            hidden_states = seasonal_hidden_states + trend_hidden_states
        else:
            hidden_states, attn_weights = self.backbone(inputs)
        hidden_states = self.flatten(hidden_states) # [batch_size, num_features, num_patches * hidden_size]
        # [batch_size, output_len, num_features]
        prediction = self.forecasting_head(hidden_states).transpose(1, 2)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class PatchTSTForClassification(nn.Module):
    """
    PatchTST for time series classification.
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.backbone = PatchTSTBackbone(config)
        self.flatten = nn.Flatten(start_dim=1)
        self.classification_head = PatchTSTHead(
            self.backbone.num_patches * config.hidden_size * config.num_features,
            config.num_classes,
            config.individual_head,
            config.num_features,
            config.head_dropout)
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last)
        self.output_attentions = config.output_attentions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of PatchTSTForForecasting.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, num_classes]
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        # [batch_size, num_features, num_patches, hidden_size]
        hidden_states, attn_weights = self.backbone(inputs)
        hidden_states = self.flatten(hidden_states) # [batch_size, num_features * num_patches * hidden_size]
        # [batch_size, num_classes]
        prediction = self.classification_head(hidden_states)
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction


class PatchTSTForReconstruction(nn.Module):
    """
    PatchTST for time series forecasting.
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.backbone = PatchTSTBackbone(config)
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecasting_head = PatchTSTHead(
            self.backbone.num_patches * config.hidden_size,
            config.input_len,
            config.individual_head,
            config.num_features,
            config.head_dropout)
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last)
        self.output_attentions = config.output_attentions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of PatchTSTForReconstruction.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, input_len, num_features]
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        # [batch_size, num_features, num_patches, hidden_size]
        hidden_states, attn_weights = self.backbone(inputs)
        hidden_states = self.flatten(hidden_states) # [batch_size, num_features, num_patches * hidden_size]
        # [batch_size, input_len, num_features]
        prediction = self.forecasting_head(hidden_states).transpose(1, 2)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if self.output_attentions:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
