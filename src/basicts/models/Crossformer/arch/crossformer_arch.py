from math import ceil

import torch
from einops import rearrange
from torch import nn

from basicts.modules.embed import PatchEmbedding

from ..config.crossformer_config import CrossformerConfig
from .crossformer_layers import (CrossformerDecoder, CrossformerDecoderLayer,
                                 CrossformerEncoder, CrossformerEncoderLayer)


class Crossformer(nn.Module):
    """
        Paper: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
        Link: https://openreview.net/forum?id=vSVLM2j9eie
        Official Code: https://github.com/Thinklab-SJTU/Crossformer
        Venue: ICLR 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: CrossformerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.patch_len = config.patch_len
        self.baseline = config.baseline

        in_pad_len = ceil(config.input_len / config.patch_len) * config.patch_len
        out_pad_len = ceil(config.output_len / config.patch_len) * config.patch_len
        in_num_patches = in_pad_len // config.patch_len
        out_num_patches = out_pad_len // config.patch_len

        # Embedding
        self.enc_embedding = PatchEmbedding(
            config.hidden_size,
            config.patch_len,
            config.patch_len,
            (in_pad_len - config.input_len, 0),
            config.dropout)
        self.pre_norm = nn.LayerNorm(config.hidden_size)

        # Encoder
        self.encoder = CrossformerEncoder(
            nn.ModuleList(
                [
                    CrossformerEncoderLayer(
                        num_features=config.num_features,
                        hidden_size=config.hidden_size,
                        n_heads=config.n_heads,
                        intermediate_size=config.intermediate_size,
                        num_patches=in_num_patches if l == 0 \
                            else ceil(in_num_patches / config.win_size ** l),
                        win_size=config.win_size if l > 0 else 1,
                        factor=config.factor,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout
                    ) for l in range(config.num_layers)
                ]
            )
        )

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, out_num_patches, config.hidden_size)
            )

        self.decoder = CrossformerDecoder(
            nn.ModuleList(
                [
                    CrossformerDecoderLayer(
                        num_features=config.num_features,
                        hidden_size=config.hidden_size,
                        n_heads=config.n_heads,
                        intermediate_size=config.intermediate_size,
                        num_patches=out_num_patches,
                        factor=config.factor,
                        hidden_act=config.hidden_act,
                        dropout=config.dropout,
                        ) for _ in range(config.num_layers + 1)
                ]
            )
        )

        self.forecasting_head = nn.ModuleList(
            [nn.Linear(
                config.hidden_size, config.patch_len
                ) for _ in range(config.num_layers + 1)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of the Crossformer model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features]
        """

        batch_size = inputs.size(0)
        base = inputs.mean(dim=1, keepdim=True) if self.baseline else 0
        # [batch_size * num_features, num_patches, hidden_size]
        hidden_states = self.enc_embedding(inputs)
        hidden_states = self.pre_norm(hidden_states)

        # Encoder
        enc_hidden_states_list = self.encoder(hidden_states)

        dec_in = self.dec_pos_embedding.expand(hidden_states.size(0), -1, -1)

        # Each decoder layer makes a prediction at a scale
        dec_hidden_states_list = self.decoder(dec_in, enc_hidden_states_list)
        prediction = base
        for idx, dec_hidden_states in enumerate(dec_hidden_states_list):
            # [batch_size * num_features, out_num_patches, patch_len]
            pred_at_scale = self.forecasting_head[idx](dec_hidden_states)
            # [batch_size, output_len + out_pad_len, num_features]
            pred_at_scale = rearrange(
                pred_at_scale, "(b c) n p -> b (n p) c", b=batch_size, p=self.patch_len)
            # cut off padding: [batch_size, output_len, num_features]
            prediction = prediction + pred_at_scale[:, :self.output_len, :]

        return prediction
