import torch
from torch import nn

from ..config.sparsetsf_config import SparseTSFConfig


class SparseTSF(nn.Module):
    """
    Paper: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters
    Official Code: https://github.com/lss-1138/SparseTSF
    Link: https://arxiv.org/abs/2405.00946
    Venue: ICML 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: SparseTSFConfig):
        super().__init__()

        # get parameters
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.period_len = config.period_len

        self.input_segs = self.input_len // self.period_len
        self.output_segs = self.output_len // self.period_len

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            padding=self.period_len // 2,
            bias=False
            )

        self.projection = nn.Linear(self.input_segs, self.output_segs, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = inputs.shape
        # normalization and permute
        seq_mean = torch.mean(inputs, dim=1, keepdim=True)
        inputs = (inputs - seq_mean).permute(0, 2, 1) # [batch_size, num_features, input_len]

        # 1D convolution aggregation
        inputs = self.conv(inputs.reshape(-1, 1, self.input_len)).reshape(-1, num_features, self.input_len) + inputs

        # downsampling:
        # [batch_size,num_features,input_len] -> [batch_size * num_features, n, w] -> [batch_size * num_features, w, n]
        inputs = inputs.reshape(-1, self.input_segs, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        prediction = self.projection(inputs)  # [batch_size * num_features, w, m]

        # upsampling:
        # [batch_size * num_features, w, m] -> [batch_size * num_features, m, w] -> [batch_size, num_features, output_len]
        prediction = prediction.permute(0, 2, 1).reshape(batch_size, num_features, self.output_len)

        # permute and denorm
        prediction = prediction.permute(0, 2, 1) + seq_mean

        return prediction
