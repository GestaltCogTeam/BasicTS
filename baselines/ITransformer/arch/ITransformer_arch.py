import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.utils import data_transformation_4_xformer

from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .embed import DataEmbedding_inverted


class ITransformer(nn.Module):
    """
    Paper:
        iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
    Link: https://arxiv.org/abs/2310.06625
    Ref Official Code:
            https://github.com/lucidrains/iTransformer
    """

    def __init__(self, **model_args):
        super(ITransformer, self).__init__()
        self.seq_len = model_args["seq_len"]
        self.label_len = int(model_args["label_len"])
        self.pred_len = model_args["pred_len"]
        self.output_attention = model_args["output_attention"]
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(model_args["seq_len"], model_args["d_model"], model_args["dropout"])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, model_args["factor"], attention_dropout= model_args["dropout"],
                                      output_attention= model_args["output_attention"]), model_args["d_model"], model_args["n_heads"]),
                    model_args["d_model"],
                    model_args["d_ff"],
                    dropout = model_args["dropout"],
                    activation = model_args["activation"]
                ) for l in range(model_args["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(model_args["d_model"])
        )
        # Decoder
        self.projection = nn.Linear(model_args["d_model"], model_args["pred_len"], bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=self.label_len)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        dec_out[:, -self.pred_len:, :]
        return dec_out.unsqueeze(-1)



