import torch
import torch.nn as nn

from .embed import DataEmbedding_wo_pos
from .auto_correlation import AutoCorrelation, AutoCorrelationLayer
from .enc_dec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from ..utils import data_transformation_4_xformer


class Autoformer(nn.Module):
    """
    Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Link: https://arxiv.org/abs/2206.09112
    Ref official code: https://github.com/thuml/Autoformer
    """

    def __init__(self, **model_args):
        super(Autoformer, self).__init__()
        assert model_args["seq_len"] % 1 == 0.0 and model_args["seq_len"] % 1 == 0.0 and model_args["pred_len"] % 1 == 0.0
        self.seq_len = int(model_args["seq_len"])
        self.label_len = int(model_args["label_len"])
        self.pred_len = int(model_args["pred_len"])
        self.output_attention = model_args['output_attention']

        # Decomp
        kernel_size = model_args["moving_avg"]
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(model_args["enc_in"], model_args["d_model"], model_args["embed"], model_args["freq"],
                                                  model_args["dropout"])
        self.dec_embedding = DataEmbedding_wo_pos(model_args["dec_in"], model_args["d_model"], model_args["embed"], model_args["freq"],
                                                  model_args["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=model_args["output_attention"]),
                        model_args["d_model"], model_args["n_heads"]),
                    model_args["d_model"],
                    model_args["d_ff"],
                    moving_avg=model_args["moving_avg"],
                    dropout=model_args["dropout"],
                    activation=model_args["activation"]
                ) for l in range(model_args["e_layers"])
            ],
            norm_layer=my_Layernorm(model_args["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=False),
                        model_args["d_model"], model_args["n_heads"]),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=False),
                        model_args["d_model"], model_args["n_heads"]),
                    model_args["d_model"],
                    model_args["c_out"],
                    model_args["d_ff"],
                    moving_avg=model_args["moving_avg"],
                    dropout=model_args["dropout"],
                    activation=model_args["activation"],
                )
                for l in range(model_args["d_layers"])
            ],
            norm_layer=my_Layernorm(model_args["d_model"]),
            projection=nn.Linear(model_args["d_model"], model_args["c_out"], bias=True)
        )

    def forward_xformer(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, N, C]

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]
        
        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data, future_data=future_data, start_token_len=self.label_len)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction
