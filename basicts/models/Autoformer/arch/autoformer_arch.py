import torch
import torch.nn as nn

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding_wo_pos, DataEmbedding
from .auto_correlation import AutoCorrelation, AutoCorrelationLayer
from .enc_dec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Link: https://arxiv.org/abs/2106.13008
    Ref Official Code: https://github.com/thuml/Autoformer
    Venue: NeurIPS 2021
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, **model_args):
        super(Autoformer, self).__init__()
        self.seq_len = int(model_args["seq_len"])
        self.label_len = int(model_args["label_len"])
        self.pred_len = int(model_args["pred_len"])
        self.output_attention = model_args["output_attention"]

        self.time_of_day_size = model_args.get("time_of_day_size", None)
        self.day_of_week_size = model_args.get("day_of_week_size", None)
        self.day_of_month_size = model_args.get("day_of_month_size", None)
        self.day_of_year_size = model_args.get("day_of_year_size", None)
        self.embed = model_args["embed"]

        # Decomp
        kernel_size = model_args["moving_avg"]
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
                                                    model_args["enc_in"],
                                                    model_args["d_model"],
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    model_args["embed"],
                                                    model_args["num_time_features"],
                                                    model_args["dropout"])
        self.dec_embedding = DataEmbedding_wo_pos(
                                                    model_args["dec_in"],
                                                    model_args["d_model"],
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    model_args["embed"],
                                                    model_args["num_time_features"],
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

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:
        """Feed forward of AutoFormer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in AutoFormer.
           See: 
            - https://github.com/thuml/Autoformer/blob/e116bbcf41f537f4ab53d172d9babfc0a026330f/layers/AutoCorrelation.py#L103
            - https://github.com/thuml/Autoformer/blob/e116bbcf41f537f4ab53d172d9babfc0a026330f/exp/exp_main.py#L136

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

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
