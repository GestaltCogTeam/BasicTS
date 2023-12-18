import torch
import torch.nn as nn

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding
from .Causal_Conv import CausalConv
from .Multi_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, MultiCorrelation
from .Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer,my_Layernorm, series_decomp


class Corrformer(nn.Module):
    """
    Paper:
        Interpretable weather forecasting for worldwide stations with a unified deep model. Nature Machine Intelligence, 2023.
    Link: https://www.nature.com/articles/s42256-023-00667-9
    Ref Official Code:
        https://github.com/thuml/Corrformer
    Note:
        In order to enable the model to run on datasets such as ETT and PEMS04, we made appropriate modifications to the spatial encoding.
        We employed the spatial encoding strategy described in the literature (https://arxiv.org/abs/2208.05233).
    """
    def __init__(self, enc_in,dec_in,c_out,seq_len,label_len,pred_len,factor_temporal,factor_spatial,
                    d_model,moving_avg,n_heads,e_layers,d_layers,d_ff,dropout, variable_num,node_num,node_list,
                    enc_tcn_layers,dec_tcn_layers,output_attention,embed,activation,
                    num_time_features,time_of_day_size,day_of_week_size,day_of_month_size,day_of_year_size):
        super(Corrformer, self).__init__()

        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.output_attention = output_attention
        self.c_out = c_out
        self.node_num = node_num
        self.node_list = node_list  # node_num = node_list[0]*node_list[1]*node_list[2]...

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.day_of_month_size = day_of_month_size
        self.day_of_year_size = day_of_year_size
        self.embed = embed

        # Decomp
        self.moving_avg = moving_avg
        self.decomp = series_decomp(self.moving_avg)

        # Encoding

        self.enc_embedding = DataEmbedding(enc_in, d_model, self.node_num, variable_num,
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    embed,
                                                    num_time_features,
                                                    dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, node_num, variable_num,
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    embed,
                                                    num_time_features,
                                                    dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor_temporal, attention_dropout=dropout,
                                            output_attention= output_attention),
                            d_model, n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=d_model // n_heads * self.seq_len,
                                    num_channels=[d_model // n_heads * self.seq_len] \
                                                 * enc_tcn_layers,
                                    kernel_size=3),
                                False, factor_spatial, attention_dropout=dropout,
                                output_attention=output_attention),
                            d_model, n_heads),
                        node_num,
                        self.node_list,
                        dropout=dropout,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=self.moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, factor_temporal, attention_dropout=dropout,
                                            output_attention=False),
                            d_model, n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=d_model // n_heads * (self.label_len + self.pred_len),
                                    num_channels=[d_model // n_heads * (self.label_len + self.pred_len)] \
                                                 * dec_tcn_layers,
                                    kernel_size=3),
                                False, factor_spatial, attention_dropout=dropout,
                                output_attention=output_attention),
                            d_model, n_heads),
                        node_num,
                        self.node_list,
                        dropout=dropout,
                    ),
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor_temporal, attention_dropout=dropout,
                                            output_attention=False),
                            d_model, n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=d_model // n_heads * (self.label_len + self.pred_len),
                                    num_channels=[d_model // n_heads * (self.label_len + self.pred_len)] \
                                                 * dec_tcn_layers,
                                    kernel_size=3),
                                False, factor_spatial, attention_dropout=dropout,
                                output_attention=output_attention),
                            d_model, n_heads),
                        node_num,
                        self.node_list,
                        dropout=dropout,
                    ),

                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=self.moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
        self.affine_weight = nn.Parameter(torch.ones(1, 1, enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=self.label_len)
        enc_self_mask = None
        dec_self_mask = None
        dec_enc_mask = None

        # init & normalization

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc * self.affine_weight.repeat(1, 1, self.node_num) + self.affine_bias.repeat(1, 1, self.node_num)

        # decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input init
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        B, L, D = x_enc.shape
        _, _, C = x_mark_enc.shape
        x_enc = x_enc.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask= enc_self_mask)
        # dec
        B, L, D = seasonal_init.shape
        _, _, C = x_mark_dec.shape
        seasonal_init = seasonal_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        trend_init = trend_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_dec = x_mark_dec.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :] \
            .view(B, self.node_num, self.pred_len, D // self.node_num).permute(0, 2, 1, 3).contiguous() \
            .view(B, self.pred_len, D)  # B L D

        # scale back
        dec_out = dec_out - self.affine_bias.repeat(1, 1, self.node_num)
        dec_out = dec_out / (self.affine_weight.repeat(1, 1, self.node_num) + 1e-10)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out.unsqueeze(-1) # [B, L, D]

