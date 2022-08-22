import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.archs.registry import ARCH_REGISTRY

"""
    Paper: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
"""

class HINetwork(nn.Module):
    def __init__(self, input_length: int, output_length: int, channel=None, reverse=False):
        """we use HI[1] as the baseline model for the pipline.
        [1] Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting

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
        """feedforward function of HI.

        Args:
            history_data (torch.Tensor): shape = [B, L_in, N, C]

        Returns:
            torch.Tensor: model prediction [B, L_out, N, C].
        """
        B, L_in, N, C = history_data.shape
        assert self.input_length == L_in, 'error input length'
        if self.channel is not None:
            history_data = history_data[..., self.channel]
        prediction = history_data[:, -self.output_length:, :, :]
        if self.reverse:
            prediction = prediction.flip(dims=[1])
        return prediction
