

import torch
import torch.nn as nn
import torch.nn.functional as F


class HINetwork(nn.Module):
    """
    Paper: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
    Link: https://arxiv.org/abs/2103.16349
    Official code: None
    Venue: CIKM 2021
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, input_length: int, output_length: int, channel=None, reverse=False):
        """
        Init HI.

        Args:
            input_length (int): input time series length
            output_length (int): prediction time series length
            channel (list, optional): selected channels. Defaults to None.
            reverse (bool, optional): if reverse the prediction of HI. Defaults to False.
        """

        super(HINetwork, self).__init__()
        assert input_length >= output_length, "HI model requires input length > output length"
        self.input_length    = input_length
        self.output_length   = output_length
        self.channel         = channel
        self.reverse         = reverse
        self.fake_param      = nn.Linear(1, 1)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function of HI.

        Args:
            history_data (torch.Tensor): shape = [B, L_in, N, C]

        Returns:
            torch.Tensor: model prediction [B, L_out, N, C].
        """

        B, L_in, N, C = history_data.shape
        assert self.input_length == L_in, 'error input length'
        if self.channel is not None:
            history_data = history_data[..., self.channel]
        # historical inertia 
        prediction = history_data[:, -self.output_length:, :, :]
        # last point
        # prediction = history_data[:, [-1], :, :].expand(-1, self.output_length, -1, -1)
        if self.reverse:
            prediction = prediction.flip(dims=[1])
        return prediction
