"""TQNet: native timestamp-query forecasting model."""

from typing import Optional
import torch
from basicts.modules.norm.forecasting.revin import ForecastRevIN
from torch import nn
from ..config.tqnet_config import TQNetConfig


class TQNet(nn.Module):
    """
    Paper: Temporal Query Network for Efficient Multivariate Time Series Forecasting
    Link: https://proceedings.mlr.press/v267/lin25e.html
    Ref Official Code: https://github.com/ACAT-SCUT/TQNet
    Venue: ICML 2025
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {
        "eps": 1e-05,
        "affine": False,
        "unbiased": True,
        "detach_mean": False,
        "detach_stdev": False,
    }

    def __init__(self, config: TQNetConfig):
        super(TQNet, self).__init__()
        self.seq_len = config.input_len
        self.pred_len = config.output_len
        self.enc_in = config.num_features
        self.cycle_len = config.cycle_len
        self.model_type = "mlp"
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.use_revin = config.use_revin
        if self.use_revin:
            self.input_normalization = ForecastRevIN(
                config.num_features, **self._input_normalization_defaults
            )
        self.use_tq = True
        self.channel_aggre = True
        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(
                torch.zeros(self.cycle_len, self.enc_in), requires_grad=True
            )
        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5
            )
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len)
        )

    def _forecast(self, x, cycle_index):
        if self.use_revin:
            x = self.input_normalization(x, "norm")
        x_input = x.permute(0, 2, 1)
        if self.use_tq:
            gather_index = (
                cycle_index.view(-1, 1)
                + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
            ) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=query_input, key=x_input, value=x_input
                )[0]
            else:
                channel_information = query_input
        elif self.channel_aggre:
            channel_information = self.channelAggregator(
                query=x_input, key=x_input, value=x_input
            )[0]
        else:
            channel_information = 0
        input = self.input_proj(x_input + channel_information)
        hidden = self.model(input)
        output = self.output_proj(hidden + input).permute(0, 2, 1)
        if self.use_revin:
            output = self.input_normalization(output, "denorm")
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        cycle_index: Optional[torch.Tensor] = None,
    ):
        if cycle_index is None:
            if inputs_timestamps is None:
                cycle_index = torch.zeros(
                    inputs.size(0), dtype=torch.long, device=inputs.device
                )
            else:
                cycle_index = torch.floor(
                    inputs_timestamps[:, -1, 0].clamp(0, 1 - 1e-06) * self.cycle_len
                ).long()
        return self._forecast(inputs, cycle_index)
