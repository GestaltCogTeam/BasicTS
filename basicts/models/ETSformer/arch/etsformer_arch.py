import torch
import torch.nn as nn
from einops import reduce

from .modules import ETSEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


class ETSformer(nn.Module):
    """
    Paper: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
    Official Code: https://github.com/salesforce/ETSformer
    Link: https://arxiv.org/abs/2202.01381
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """
    def __init__(self,  **model_args):
        super().__init__()
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.enc_in = model_args['enc_in']
        self.d_model = model_args['d_model']
        self.dropout = model_args['dropout']
        self.n_head = model_args['n_heads']
        self.c_out = model_args['c_out']
        self.K = model_args['K']
        self.d_ff = model_args['d_ff']
        self.sigma = model_args['sigma']
        self.activation = model_args['activation']
        self.output_attention = model_args['output_attention']

        assert self.e_layers == self.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = ETSEmbedding(self.enc_in, self.d_model, dropout=self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.d_model, self.n_head, self.c_out, self.seq_len, self.pred_len, self.K,
                    dim_feedforward=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    output_attention=self.output_attention,
                ) for _ in range(self.e_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self.d_model, self.n_head, self.c_out, self.pred_len,
                    dropout=self.dropout,
                    output_attention=self.output_attention,
                ) for _ in range(self.d_layers)
            ],
        )

        self.transform = Transform(sigma=self.sigma)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                enc_self_mask=None,
                decomposed=False, attention=False):
        """
                Args:
                    history_data (Tensor): Input data with shape: [B, L1, N, C]
                    future_data (Tensor): Future data with shape: [B, L2, N, C]

                Returns:
                    torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        x_enc = history_data[:,:,:,0]
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc)
        level, growths, seasons, season_attns, growth_attns = self.encoder(res, x_enc, attn_mask=enc_self_mask)

        growth, season, growth_dampings = self.decoder(growths, seasons)

        if decomposed:
            return level[:, -1:], growth, season

        preds = level[:, -1:] + growth + season

        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum('bth,oh->bhot', [growth_attn.squeeze(-1), growth_damping]))

            season_attns = torch.stack(season_attns, dim=0)[:, :, -self.pred_len:]
            season_attns = reduce(season_attns, 'l b d o t -> b o t', reduction='mean')
            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len:]
            decoder_growth_attns = reduce(decoder_growth_attns, 'l b d o t -> b o t', reduction='mean')
            return preds, season_attns, decoder_growth_attns
        preds = preds.unsqueeze(-1)
        return preds
