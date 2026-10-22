"""S_Mamba forecasting architecture."""

from basicts.modules.embed.time_features import calendar_features
from ..config.s_mamba_config import SMambaConfig
import torch
from basicts.modules.norm.forecasting.revin import ForecastRevIN
import torch.nn as nn
from .Mamba_EncDec import Encoder, EncoderLayer
from basicts.modules.embed import SequenceEmbedding
from basicts.modules.ssm import MambaMixer as Mamba


class SMamba(nn.Module):
    """
    Paper: Is Mamba Effective for Time Series Forecasting?
    Link: https://doi.org/10.1016/j.neucom.2024.129178
    Ref Official Code: https://github.com/wzhwzhwzh0921/S-D-Mamba
    Venue: Neurocomputing 2025
    Task: Long-term Time Series Forecasting

    Note:
        This port uses the pure-PyTorch mambapy mixer in place of the fused Mamba kernel.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {
        "eps": 1e-05,
        "affine": False,
        "detach_stdev": False,
        "center_before_variance": True,
    }

    def __init__(self, config: SMambaConfig):
        super(SMamba, self).__init__()
        self.embed = "timeF"
        self.freq = "h"
        self.pred_len = config.output_len
        self.seq_len = config.input_len
        self.use_norm = config.use_norm
        if self.use_norm:
            self.input_normalization = ForecastRevIN(
                config.num_features, **self._input_normalization_defaults
            )
        self.d_ff = config.intermediate_size
        self.d_model = config.hidden_size
        self.d_state = config.state_size
        self.dropout = config.dropout
        self.activation = config.hidden_act
        self.e_layers = config.num_layers
        self.enc_embedding = SequenceEmbedding(self.seq_len, self.d_model, self.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=self.d_model, d_state=self.d_state, d_conv=2, expand=1
                    ),
                    Mamba(
                        d_model=self.d_model, d_state=self.d_state, d_conv=2, expand=1
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)
        self.time_of_day_size = config.time_of_day_size
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)
        self.input_length = config.input_len

    def forward_xformer(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_norm:
            x_enc = self.input_normalization(x_enc, "norm")
        (_, _, N) = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        (enc_out, attns) = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
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
        if marks.shape[-1]:
            marks[..., 0] = (
                marks[..., 0]
                * self.time_of_day_size
                // (self.time_of_day_size / 24)
                / 23.0
            )
        return self.forward_xformer(inputs.float(), marks - 0.5, None, None)
