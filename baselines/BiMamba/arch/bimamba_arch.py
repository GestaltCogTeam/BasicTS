import torch
import torch.nn as nn
import numpy as np
from .layers import Encoder, EncoderLayer, Flatten_Head
from .Embed import PatchEmbedding, TruncateModule
from einops import rearrange
from mamba_ssm import Mamba
from basicts.utils import data_transformation_4_xformer

from argparse import Namespace
import pdb 


class BiMamba(nn.Module):
    """
    Paper: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting
    Link: https://arxiv.org/abs/2404.15772
    Official Code: https://github.com/Leopold2333/Bi-Mamba4TS
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(BiMamba, self).__init__()
        
        configs = Namespace(**model_args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.embed_type = configs.embed_type
        if configs.SRA:
            self.ch_ind = self.SRA(configs.corr , configs.threshold)
        else:
            self.ch_ind = configs.ch_ind

        # patching
        # being able to divide indicates that there is no need to forcefully padding
        if configs.seq_len % configs.stride==0:
            self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
            process_layer = nn.Identity()
        else:
            # padding at tail
            if configs.padding_patch=="end":
                padding_length = configs.stride - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 2)
                process_layer = nn.ReplicationPad1d((0, padding_length))
            # if not padding, then execute cutting
            else:
                truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
                process_layer = TruncateModule(truncated_length)
        self.local_token_layer = PatchEmbedding(configs.seq_len, configs.d_model, configs.patch_len, configs.stride, configs.dropout, process_layer,
                                                pos_embed_type=None if configs.embed_type in [0, 2] else configs.pos_embed_type,
                                                learnable=configs.pos_learnable,
                                                ch_ind=configs.ch_ind)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model, 
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=configs.d_state, 
                          d_conv=configs.d_conv, 
                          expand=configs.e_fact,
                          use_fast_path=True),
                    Mamba(d_model=configs.d_model, 
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=configs.d_state, 
                          d_conv=configs.d_conv, 
                          expand=configs.e_fact,
                          use_fast_path=True),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    bi_dir=configs.bi_dir,
                    residual=configs.residual==1
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        head_nf = configs.d_model * self.patch_num
        self.head = Flatten_Head(False, configs.enc_in, head_nf, self.pred_len)

    def SRA(self, corr, threshold):
        high_corr_matrix = corr >= threshold
        num_high_corr = np.maximum(high_corr_matrix.sum(axis=1)-1, 0)

        positive_corr_matrix = corr >= 0
        num_positive_corr = np.maximum(positive_corr_matrix.sum(axis=1)-1, 0)
        max_high_corr = num_high_corr.max()
        max_positive_corr = num_positive_corr.max()
        r = max_high_corr/max_positive_corr
        print('SRA -> channel mixing' if r >= 1 - threshold else 'channel independent')
        return 0 if r >= 1 - threshold else 1

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        if self.revin:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # B L M
        B, _, M = x_enc.shape

        # [B, M, L] -> [B*M, N, D] or [B*(M+4), N, D]
        enc_out, _ = self.local_token_layer(x_enc.permute(0, 2, 1),
                                            x_mark_enc.permute(0, 2, 1) if self.embed_type==2 else None)
        if not self.ch_ind:
            enc_out = rearrange(enc_out, '(B M) N D -> (B N) M D', B=B)

        enc_out = self.encoder(enc_out)
        # output: [B*M, N, D] or [B*N, M, D] -> [B x M x H]
        if not self.ch_ind:
            dec_out = rearrange(enc_out, '(B N) M D -> B M N D', B=B)
        else:
            dec_out = rearrange(enc_out, '(B M) N D -> B M N D', B=B)
        dec_out = self.head(dec_out).permute(0, 2, 1)[:, :, :M]

        if self.revin:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        # history_data[..., 1] = history_data[..., 1] * self.time_of_day_size // (self.time_of_day_size / 24) / 23.0
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0)
        #print(x_mark_enc.shape, x_mark_dec.shape)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction.unsqueeze(-1)