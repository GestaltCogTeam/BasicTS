import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding_wo_pos, DataEmbedding
from .auto_correlation import AutoCorrelationLayer
from .fourier_correlation import FourierBlock, FourierCrossAttention
from .multi_wavelet_correlation import MultiWaveletCross, MultiWaveletTransform
from .fedformer_enc_dec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi


class FEDformer(nn.Module):
    """
    Paper: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting
    Link: https://arxiv.org/abs/2201.12740v3
    Ref Official Code: https://github.com/MAZiqing/FEDformer
    Venue: ICML 2022
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(FEDformer, self).__init__()
        self.version = model_args["version"]
        self.mode_select = model_args["mode_select"]
        self.modes = model_args["modes"]
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

        # Decomp
        kernel_size = model_args["moving_avg"]
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
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

        if model_args["version"] == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=model_args["d_model"], L=model_args["L"], base=model_args["base"])
            decoder_self_att = MultiWaveletTransform(
                ich=model_args["d_model"], L=model_args["L"], base=model_args["base"])
            decoder_cross_att = MultiWaveletCross(in_channels=model_args["d_model"],
                                                  out_channels=model_args["d_model"],
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=model_args["modes"],
                                                  ich=model_args["d_model"],
                                                  base=model_args["base"],
                                                  activation=model_args["cross_activation"])
        else:
            encoder_self_att = FourierBlock(in_channels=model_args["d_model"],
                                            out_channels=model_args["d_model"],
                                            seq_len=self.seq_len,
                                            modes=model_args["modes"],
                                            mode_select_method=model_args["mode_select"])
            decoder_self_att = FourierBlock(in_channels=model_args["d_model"],
                                            out_channels=model_args["d_model"],
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=model_args["modes"],
                                            mode_select_method=model_args["mode_select"])
            decoder_cross_att = FourierCrossAttention(in_channels=model_args["d_model"],
                                                      out_channels=model_args["d_model"],
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=model_args["modes"],
                                                      mode_select_method=model_args["mode_select"])
        # Encoder
        enc_modes = int(min(model_args["modes"], model_args["seq_len"]//2))
        dec_modes = int(
            min(model_args["modes"], (model_args["seq_len"]//2+model_args["pred_len"])//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
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
                        decoder_self_att,
                        model_args["d_model"], model_args["n_heads"]),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
        """Feed forward of FEDformer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in FEDFormer.

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
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                            x_dec.shape[2]]).to(mean.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
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

