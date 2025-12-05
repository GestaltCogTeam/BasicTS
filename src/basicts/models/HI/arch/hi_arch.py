

from typing import List

import torch
from torch import nn

from basicts.runners.callback import NoBP

from ..config.hi_config import HIConfig


class HI(nn.Module):
    """
    Paper: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
    Link: https://arxiv.org/abs/2103.16349
    Official code: None
    Venue: CIKM 2021
    Task: Long-term Time Series Forecasting
    """

    _required_callbacks: List[type] = [NoBP]

    def __init__(self, config: HIConfig):
        """
        Init HI.

        Args:
            config (HIConfig): model config.
        """

        super().__init__()
        self.input_len    = config.input_len
        self.output_len   = config.output_len
        assert self.input_len >= self.output_len, "HI model requires input length > output length"
        self.reverse         = config.reverse
        self.fake_param      = nn.Linear(1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward function of HI.

        Args:
            inputs (torch.Tensor): shape = [B, L_in, N]

        Returns:
            torch.Tensor: model prediction [B, L_out, N].
        """

        # historical inertia
        prediction = inputs[:, -self.output_len:, :]
        # last point
        # prediction = inputs[:, [-1], :].expand(-1, self.output_len, -1)
        if self.reverse:
            prediction = prediction.flip(dims=[1])
        return prediction
