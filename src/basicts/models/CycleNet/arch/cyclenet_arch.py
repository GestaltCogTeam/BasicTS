import torch
import torch.nn as nn
from ..config.cyclenet_config import CycleNetConfig

from basicts.modules.norm import RevIN
from basicts.modules import MLPLayer


class RecurrentCycle(nn.Module):
    """
    From the author:
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    """

    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size), requires_grad=True
        )

    def forward(self, index, length):
        # index: [B]
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index.long()]  # [B, length, channel_size]


class CycleNet(nn.Module):
    """
    Paper: CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns
    Link: https://arxiv.org/abs/2409.18479
    Ref Official Code: https://github.com/ACAT-SCUT/CycleNet
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-06, "affine": False}

    def __init__(self, config: CycleNetConfig):
        super().__init__()

        self.input_len = config.input_len
        self.output_len = config.output_len

        self.cycle_len = config.cycle_len
        self.cycle_pattern = config.cycle_pattern

        self.model_type = config.model_type
        self.use_revin = config.use_revin

        # cyclic embedding table
        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len, channel_size=config.num_features
        )

        # ---------------------------------------------------------
        # backbone: linear or MLP, applied independently per-channel
        # ---------------------------------------------------------
        assert config.model_type in ["linear", "mlp"]

        if config.model_type == "linear":
            self.model = nn.Linear(self.input_len, self.output_len)
        else:
            self.model = MLPLayer(
                input_size=config.input_len,
                intermediate_size=config.hidden_size,
                output_size=config.output_len,
                hidden_act=config.hidden_act,
                dropout=config.dropout,
            )

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=False)

    def forward(
        self,
        inputs: torch.Tensor,  # [B, L, N, 1]
        inputs_timestamps: torch.Tensor,  # [B, L, N, C_ts]
    ) -> torch.Tensor:
        """
        Returns:
            [B, pred_len, N, 1]
        """
        # ---------------------------------------------------------
        # 1. extract cycle index from timestamps
        #    (replace original history[...,1:])
        # ---------------------------------------------------------
        if self.cycle_pattern == "daily":
            ts_daily = inputs_timestamps[..., 0]  # [B,L,N]
            cycle_index = (ts_daily * self.cycle_len)[:, -1]  # [B]

        elif self.cycle_pattern == "daily&weekly":
            ts_daily = inputs_timestamps[..., 0]
            ts_week = inputs_timestamps[..., 1]
            cycle_index = (ts_daily * self.cycle_len * 7 + ts_week * 7)[:, -1]  # [B]

        else:
            raise ValueError("cycle_pattern must be 'daily' or 'daily&weekly'")

        # ---------------------------------------------------------
        # 2. RevIN-like instance normalization
        # ---------------------------------------------------------
        # ---- Instance-wise normalization ----
        if self.use_revin:
            x = self.revin(inputs, "norm")
        else:
            x = inputs

        # ---------------------------------------------------------
        # 3. remove cycle from input
        # ---------------------------------------------------------
        cycle_in = self.cycleQueue(cycle_index, self.input_len)  # [B, L, C_in]
        x = x - cycle_in  # broadcast: [B,L,N]

        # ---------------------------------------------------------
        # 4. forecasting (channel independent)
        # ---------------------------------------------------------
        # input for model: [B,N,L] → out: [B,N,pred_len]
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # ---------------------------------------------------------
        # 5. add cycle back to prediction
        # ---------------------------------------------------------
        cycle_out = self.cycleQueue(
            (cycle_index + self.input_len) % self.cycle_len, self.output_len
        )  # [B,pred_len,C_in]

        y = y + cycle_out  # [B,pred_len,N]

        # ---------------------------------------------------------
        # 6. de-normalization
        # ---------------------------------------------------------
        if self.use_revin:
            prediction = self.revin(y, "denorm")
        else:
            prediction = y

        # final shape [B,pred_len,N,1]
        return prediction
