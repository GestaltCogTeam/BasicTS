

from typing import List

import torch
import torch.nn as nn

from basicts.runners.callback import NoBP


class HINetwork(nn.Module):
    """
    Paper: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
    Link: https://arxiv.org/abs/2103.16349
    Official code: None
    Venue: CIKM 2021
    Task: Long-term Time Series Forecasting
    """

    _required_callbacks: List[type] = [NoBP]

    def __init__(self, input_length: int, output_length: int, reverse=False):
        """
        Init HI.

        Args:
            input_length (int): input time series length
            output_length (int): prediction time series length
            reverse (bool, optional): if reverse the prediction of HI. Defaults to False.
        """

        super(HINetwork, self).__init__()
        assert input_length >= output_length, "HI model requires input length > output length"
        self.input_length    = input_length
        self.output_length   = output_length
        self.reverse         = reverse
        self.fake_param      = nn.Linear(1, 1, bias=False)

    # pylint: disable=unused-argument
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function of HI.

        Args:
            history_data (torch.Tensor): shape = [B, L_in, N]

        Returns:
            torch.Tensor: model prediction [B, L_out, N].
        """

        _, L_in, _ = inputs.shape
        assert self.input_length == L_in, "error input length"
        # historical inertia
        prediction = inputs[:, -self.output_length:, :]
        # last point
        # prediction = history_data[:, [-1], :].expand(-1, self.output_length, -1)
        if self.reverse:
            prediction = prediction.flip(dims=[1])
        return prediction
