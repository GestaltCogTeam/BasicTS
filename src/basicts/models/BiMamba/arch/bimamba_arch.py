"""BiMamba forecasting architecture."""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from basicts.modules.embed.time_features import calendar_features
from basicts.modules.norm.forecasting.revin import ForecastRevIN
from basicts.modules.ssm import MambaMixer as Mamba

from ..config.bimamba_config import BiMambaConfig
from .Embed import PatchEmbedding, TruncateModule
from .layers import Encoder, EncoderLayer, Flatten_Head


class BiMamba(nn.Module):
    """
    Paper: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting
    Link: https://arxiv.org/abs/2404.15772
    Ref Official Code: https://github.com/Leopold2333/Bi-Mamba4TS
    Venue: arXiv 2024 (preprint)
    Task: Long-term Time Series Forecasting

    Note:
        This port uses standard mambapy Mamba blocks rather than the paper's
        Mamba+ blocks; it is not an exact reproduction of the authors' implementation.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {
        "eps": 1e-05,
        "affine": False,
        "detach_stdev": False,
        "center_before_variance": True,
    }

    def __init__(self, config: BiMambaConfig):
        super(BiMamba, self).__init__()
        self.seq_len = config.input_len
        self.pred_len = config.output_len
        self.revin = config.use_revin
        if self.revin:
            self.input_normalization = ForecastRevIN(
                config.num_features, **self._input_normalization_defaults
            )
        self.embed_type = config.embed_type
        if config.use_sra:
            self.ch_ind = self.SRA(config.correlation, config.correlation_threshold)
        else:
            self.ch_ind = int(config.channel_independence)
        if config.input_len % config.stride == 0:
            self.patch_num = int(
                (config.input_len - config.patch_len) / config.stride + 1
            )
            process_layer = nn.Identity()
        elif config.padding_patch == "end":
            padding_length = config.stride - config.input_len % config.stride
            self.patch_num = int(
                (config.input_len - config.patch_len) / config.stride + 2
            )
            process_layer = nn.ReplicationPad1d((0, padding_length))
        else:
            truncated_length = config.input_len - config.input_len % config.stride
            self.patch_num = int(
                (config.input_len - config.patch_len) / config.stride + 1
            )
            process_layer = TruncateModule(truncated_length)
        self.local_token_layer = PatchEmbedding(
            config.input_len,
            config.hidden_size,
            config.patch_len,
            config.stride,
            config.dropout,
            process_layer,
            pos_embed_type=None
            if config.embed_type in [0, 2]
            else config.position_embedding_type,
            learnable=config.position_learnable,
            ch_ind=int(config.channel_independence),
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=config.hidden_size,
                        d_state=config.state_size,
                        d_conv=config.conv_size,
                        expand=config.expand_factor,
                        use_fast_path=True,
                    ),
                    Mamba(
                        d_model=config.hidden_size,
                        d_state=config.state_size,
                        d_conv=config.conv_size,
                        expand=config.expand_factor,
                        use_fast_path=True,
                    ),
                    config.hidden_size,
                    config.intermediate_size,
                    dropout=config.dropout,
                    activation=config.hidden_act,
                    bi_dir=config.bidirectional,
                    residual=int(config.residual) == 1,
                )
                for _ in range(config.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.hidden_size),
        )
        head_nf = config.hidden_size * self.patch_num
        self.head = Flatten_Head(False, config.num_features, head_nf, self.pred_len)
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)
        self.input_length = config.input_len

    def SRA(self, corr, threshold):
        high_corr_matrix = corr >= threshold
        num_high_corr = np.maximum(high_corr_matrix.sum(axis=1) - 1, 0)
        positive_corr_matrix = corr >= 0
        num_positive_corr = np.maximum(positive_corr_matrix.sum(axis=1) - 1, 0)
        max_high_corr = num_high_corr.max()
        max_positive_corr = num_positive_corr.max()
        r = max_high_corr / max_positive_corr
        print("SRA -> channel mixing" if r >= 1 - threshold else "channel independent")
        return 0 if r >= 1 - threshold else 1

    def forward_xformer(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        if self.revin:
            x_enc = self.input_normalization(x_enc, "norm")
        (B, _, M) = x_enc.shape
        (enc_out, _) = self.local_token_layer(
            x_enc.permute(0, 2, 1),
            x_mark_enc.permute(0, 2, 1) if self.embed_type == 2 else None,
        )
        if not self.ch_ind:
            enc_out = rearrange(enc_out, "(B M) N D -> (B N) M D", B=B)
        enc_out = self.encoder(enc_out)
        if not self.ch_ind:
            dec_out = rearrange(enc_out, "(B N) M D -> B M N D", B=B)
        else:
            dec_out = rearrange(enc_out, "(B M) N D -> B M N D", B=B)
        dec_out = self.head(dec_out).permute(0, 2, 1)[:, :, :M]
        if self.revin:
            dec_out = self.input_normalization(dec_out, "denorm")
        return dec_out

    def forward(self, inputs, inputs_timestamps=None, targets_timestamps=None):
        marks = calendar_features(
            inputs_timestamps,
            inputs,
            self.input_length,
            self.num_timestamps,
            center=False,
        )
        return self.forward_xformer(inputs.float(), marks - 0.5, None, None)
