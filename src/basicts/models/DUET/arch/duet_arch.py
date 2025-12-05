from typing import Any, Dict, List

import torch
from einops import rearrange
from torch import nn

from basicts.modules.mlps import MLPLayer
from basicts.modules.transformer import (Encoder, EncoderLayer,
                                         MultiHeadAttention)
from basicts.runners.callback import AddAuxiliaryLoss

from ..config.duet_config import DUETConfig
from .linear_extractor_cluster import LinearExtractorCluster
from .mahalanobis_mask import MahalanobisMask


class DUET(nn.Module):

    """
    Paper: DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2412.10859
    Official Code: https://github.com/decisionintelligence/DUET
    Venue: KDD 2025
    Task: Long-term Time Series Forecasting
    """

    _required_callbacks :List[type] = [AddAuxiliaryLoss]

    def __init__(self, config: DUETConfig):
        super().__init__()
        self.cluster = LinearExtractorCluster(config)
        self.loss_coef = config.loss_coef
        self.channel_independence = config.channel_independence
        self.num_features = config.num_features
        self.mask_generator = MahalanobisMask(config.input_len)
        self.channel_transformer = Encoder(
            nn.ModuleList(
                [
                    EncoderLayer(
                        MultiHeadAttention(
                            config.hidden_size,
                            config.n_heads,
                            dropout=config.dropout,
                        ),
                        MLPLayer(
                            config.hidden_size,
                            config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dropout=config.dropout,
                        ),
                        layer_norm=(nn.LayerNorm, config.hidden_size),
                        norm_position="post"
                    )
                    for _ in range(config.num_layers)
                ]),
            layer_norm=nn.LayerNorm(config.hidden_size)
            )

        self.linear_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.output_len),
            nn.Dropout(config.fc_dropout)
            )

    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:

        """
        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_len, num_features].

        Returns:
            Dict[str, Any]: A dictionary containing:
            "prediction": the prediction tensor with shape [batch_size, output_len, num_features],
            "L_importance": the importance tensor with shape [batch_size, num_features].
        """

        if self.channel_independence:
            channel_independent_input = rearrange(inputs, "b l n -> (b n) l 1")
            hidden_states, load_balance_loss = self.cluster(channel_independent_input)
            hidden_states = rearrange(hidden_states, "(b n) l 1 -> b l n", b=inputs.shape[0])

        else:
            hidden_states, load_balance_loss = self.cluster(inputs)

        hidden_states = rearrange(hidden_states, "b d n -> b n d")
        if self.num_features > 1:
            changed_input = rearrange(inputs, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, _ = self.channel_transformer(
                hidden_states=hidden_states, attention_mask=channel_mask)
            prediction = self.linear_head(channel_group_feature)
        else:
            prediction = self.linear_head(hidden_states)

        prediction = rearrange(prediction, "b n d -> b d n")
        prediction = self.cluster.revin(prediction, "denorm")
        return {"prediction": prediction, "aux_loss": self.loss_coef * load_balance_loss}
