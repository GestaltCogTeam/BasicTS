"""ETSformer forecasting architecture."""

from ..config.etsformer_config import ETSformerConfig
import torch
import torch.nn as nn
from einops import reduce
from basicts.modules.embed.calendar_embed import CausalFeatureEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + torch.randn(x.shape).to(x.device) * self.sigma

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + torch.randn(x.size(-1)).to(x.device) * self.sigma


class ETSformer(nn.Module):
    """
    Paper: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
    Link: https://arxiv.org/abs/2202.01381
    Ref Official Code: https://github.com/salesforce/ETSformer
    Venue: arXiv 2022 (preprint)
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: ETSformerConfig):
        super().__init__()
        self.seq_len = config.input_len
        self.pred_len = config.output_len
        self.e_layers = config.num_encoder_layers
        self.d_layers = config.num_decoder_layers
        self.enc_in = config.num_features
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.n_head = config.n_heads
        self.c_out = config.num_features
        self.K = config.top_k
        self.d_ff = config.intermediate_size
        self.sigma = config.sigma
        self.activation = config.hidden_act
        self.output_attention = config.output_attentions
        assert self.e_layers == self.d_layers, (
            "Encoder and decoder layers must be equal"
        )
        self.enc_embedding = CausalFeatureEmbedding(
            self.enc_in, self.d_model, dropout=self.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.d_model,
                    self.n_head,
                    self.c_out,
                    self.seq_len,
                    self.pred_len,
                    self.K,
                    dim_feedforward=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    output_attention=self.output_attention,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self.d_model,
                    self.n_head,
                    self.c_out,
                    self.pred_len,
                    dropout=self.dropout,
                    output_attention=self.output_attention,
                )
                for _ in range(self.d_layers)
            ]
        )
        self.transform = Transform(sigma=self.sigma)
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def forward(self, inputs):
        x_enc = inputs
        if self.training:
            # Random perturbations are constants; keep gradients to an optional
            # input normalization/APT adapter through this affine augmentation.
            x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc)
        (level, growths, seasons, season_attns, growth_attns) = self.encoder(
            res, x_enc, attn_mask=None
        )
        (growth, season, growth_dampings) = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds
