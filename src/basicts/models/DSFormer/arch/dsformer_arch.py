"""DSFormer forecasting architecture."""

import torch
from torch import nn

from basicts.modules.norm import RevIN

from ..config.dsformer_config import DSFormerConfig
from .decoder_block import TVADE_block
from .embed_block import embed
from .tva_block import TVA_block_att


class DSFormer(nn.Module):
    """
    Paper: DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction
    Link: https://arxiv.org/abs/2308.03274
    Ref Official Code: https://github.com/ChengqingYu/DSformer
    Venue: CIKM 2023
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: DSFormerConfig):
        super(DSFormer, self).__init__()
        Input_len = config.input_len
        out_len = config.output_len
        num_id = config.num_features
        num_layer = config.num_layers
        dropout = config.dropout
        muti_head = config.n_heads
        num_samp = config.num_samp
        IF_node = config.use_node_embedding
        IF_REVIN = config.use_revin
        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp
        self.IF_REVIN = IF_REVIN
        if IF_REVIN:
            self.RevIN = RevIN(num_id, eps=1e-5)
        self.embed_layer = embed(Input_len, num_id, num_samp, IF_node)
        self.encoder = TVA_block_att(
            self.inputlen, num_id, num_layer, dropout, muti_head, num_samp
        )
        self.laynorm = nn.LayerNorm([self.inputlen])
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(
            in_channels=self.inputlen, out_channels=out_len, kernel_size=1
        )
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def forward(self, inputs) -> torch.Tensor:
        if self.IF_REVIN:
            x = inputs
            x = self.RevIN(x, "norm").transpose(-2, -1)
        else:
            x = inputs.transpose(-2, -1)
        (x_1, x_2) = self.embed_layer(x)
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)
        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))
        if self.IF_REVIN:
            x = self.RevIN(x, "denorm")
        return x
