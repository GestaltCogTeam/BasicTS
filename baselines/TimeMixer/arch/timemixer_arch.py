import torch
import torch.nn as nn
import torch.nn.functional as F
from .Autoformer_EncDec import series_decomp
from .Embed import DataEmbedding_wo_pos
from .StandardNorm import Normalize
from basicts.utils import data_transformation_4_xformer

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, **model_args):
        super(MultiScaleSeasonMixing, self).__init__()
        self.seq_len = model_args['seq_len']
        self.down_sampling_window = model_args['down_sampling_window']
        self.down_sampling_layers = model_args['down_sampling_layers']
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(self.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, **model_args):
        super(MultiScaleTrendMixing, self).__init__()
        self.seq_len = model_args['seq_len']
        self.down_sampling_window = model_args['down_sampling_window']
        self.down_sampling_layers = model_args['down_sampling_layers']

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(self.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, **model_args):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.moving_avg = model_args['moving_avg']
        self.down_sampling_window = model_args['down_sampling_window']
        self.down_sampling_layers = model_args['down_sampling_layers']
        self.channel_independence = model_args['channel_independence']
        self.d_model = model_args['d_model']
        self.d_ff = model_args['d_ff']
        self.dp = model_args['dropout']

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dp)
        self.decomp_method = model_args['decomp_method']
        self.top_k = model_args['top_k']
        if self.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(self.moving_avg)
        elif self.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(self.top_k)
        else:
            raise ValueError('decompsition is error')

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=self.d_model, out_features=self.d_ff),
                nn.GELU(),
                nn.Linear(in_features=self.d_ff, out_features=self.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(**model_args)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(**model_args)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, **model_args):
        super(TimeMixer, self).__init__()

        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.e_layers = model_args['e_layers']
        self.moving_avg = model_args['moving_avg']
        self.down_sampling_window = model_args['down_sampling_window']
        self.down_sampling_layers = model_args['down_sampling_layers']
        self.channel_independence = model_args['channel_independence']
        self.d_model = model_args['d_model']
        self.dec_in = model_args['dec_in']
        self.enc_in = model_args['enc_in']
        self.c_out = model_args['c_out']
        self.freq = model_args['freq']
        self.dropout = model_args['dropout']
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(**model_args)
                                         for _ in range(self.e_layers)])
        self.down_sampling_method = model_args["down_sampling_method"]
        self.preprocess = series_decomp(self.moving_avg)
        self.embed = model_args['embed']
        self.enc_in = model_args['enc_in']
        self.use_norm = model_args['use_norm']
        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, self.d_model, self.embed, self.freq,
                                                      self.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                      self.dropout)

        self.layer = self.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=True if self.use_norm == 0 else False)
                for i in range(self.down_sampling_layers + 1)
            ]
        )


        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(
                    self.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                self.d_model, self.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** i),
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )



    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0)

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out.unsqueeze(-1)

