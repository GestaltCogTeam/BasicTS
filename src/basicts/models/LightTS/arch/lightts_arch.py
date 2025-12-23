import torch
from torch import nn

from basicts.modules import MLPLayer

from ..config.lightts_config import LightTSConfig


class IEBlock(nn.Module):
    """
    IEBlock in LightTS.
    """
    def __init__(
            self,
            input_size: int,
            intermediate_size: int,
            output_size: int,
            num_features: int):
        super().__init__()

        self.spatial_proj = MLPLayer(
            input_size,
            intermediate_size,
            intermediate_size // 4,
            hidden_act="leaky_relu"
            )

        self.channel_proj = nn.Linear(num_features, num_features)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(intermediate_size // 4, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class LightTS(nn.Module):
    """
    Paper: Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP
    Code (Unofficial): https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py
    Link: https://arxiv.org/abs/2207.01186
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, config: LightTSConfig):
        super().__init__()
        self.chunk_size = config.chunk_size
        assert config.input_len % self.chunk_size == 0, "`input_len` must be divisible by `chunk_size`"
        self.num_chunks = config.input_len // self.chunk_size

        self.ie_block_c = IEBlock(
            input_size=self.chunk_size,
            intermediate_size=config.hidden_size // 4,
            output_size=config.hidden_size // 4,
            num_features=self.num_chunks
        )

        self.chunk_proj_c = nn.Linear(self.num_chunks, 1)

        self.ie_block_i = IEBlock(
            input_size=self.chunk_size,
            intermediate_size=config.hidden_size // 4,
            output_size=config.hidden_size // 4,
            num_features=self.num_chunks
        )

        self.chunk_proj_i = nn.Linear(self.num_chunks, 1)

        self.ie_block = IEBlock(
            input_size=config.hidden_size // 2,
            intermediate_size=config.hidden_size // 2,
            output_size=config.output_len,
            num_features=config.num_features
        )

        self.ar = nn.Linear(config.input_len, config.output_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LightTS.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_len, num_features].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, output_len, num_features].
        """
        batch_size, _, num_features = inputs.size()
        shortcut = self.ar(inputs.transpose(1, 2)).transpose(1, 2)

        # continuous sampling
        x_conti = inputs.reshape(batch_size, self.num_chunks, self.chunk_size, num_features)
        x_conti = x_conti.permute(0, 3, 2, 1)
        x_conti = x_conti.reshape(-1, self.chunk_size, self.num_chunks)
        x_conti = self.ie_block_c(x_conti)
        x_conti = self.chunk_proj_c(x_conti).squeeze(-1)

        # interval sampling
        x_inter = inputs.reshape(batch_size, self.chunk_size, self.num_chunks, num_features)
        x_inter = x_inter.permute(0, 3, 1, 2)
        x_inter = x_inter.reshape(-1, self.chunk_size, self.num_chunks)
        x_inter = self.ie_block_i(x_inter)
        x_inter = self.chunk_proj_i(x_inter).squeeze(-1)

        x = torch.cat([x_conti, x_inter], dim=-1)

        x = x.reshape(batch_size, num_features, -1)
        prediction = self.ie_block(x.transpose(1, 2))
        prediction = prediction + shortcut
        return prediction
