import torch
from torch import nn

from basicts.modules.norm import RevIN

from ..config.mtsmixer_config import MTSMixerConfig
from .mtsmixer_layers import ChannelProjection, MixerLayer


class MTSMixer(nn.Module):
    """
    Paper: MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing
    Official Code: https://github.com/plumprc/MTS-Mixers
    Link: https://arxiv.org/abs/2302.04501
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """
    def __init__(self,  config: MTSMixerConfig):
        super().__init__()
        self.mixing_layers = nn.ModuleList(
            [MixerLayer(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.num_features) if config.use_layer_norm else None
        self.projection = ChannelProjection(
            config.input_len, config.output_len, config.num_features, config.individual)
        self.revin = RevIN(config.num_features) if config.use_revin else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MTSMixer model.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, output_len, num_features]
        """
        if self.revin:
            inputs = self.revin(inputs, "norm")

        for layer in self.mixing_layers:
            inputs = layer(inputs)

        if self.layer_norm:
            inputs = self.layer_norm(inputs)
        prediction = self.projection(inputs)

        if self.revin:
            prediction = self.revin(prediction, "denorm")

        return prediction
