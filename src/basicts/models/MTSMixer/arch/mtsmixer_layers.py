import torch
from torch import nn

from basicts.modules import MLPLayer

from ..config.mtsmixer_config import MTSMixerConfig


class FactorizedTemporalMixing(nn.Module):

    """
    Factorized Temporal Mixing Layer
    """

    def __init__(self,
                 input_size: int,
                 intermediate_size: int,
                 down_sampling: int):
        super().__init__()

        assert down_sampling in [1, 2, 3, 4, 6, 8, 12], \
            "down_sampling must be in [1, 2, 3, 4, 6, 8, 12]"
        self.down_sampling = down_sampling
        self.temporal_fac = nn.ModuleList([
            MLPLayer(
                input_size // down_sampling, intermediate_size, hidden_act="gelu"
                ) for _ in range(down_sampling)
        ])

    def merge(self, shape: torch.Size, x_list: list[torch.Tensor]) -> torch.Tensor:
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.down_sampling] = x_pad
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.down_sampling]))
        x = self.merge(x.shape, x_samp)
        return x


class ChannelProjection(nn.Module):
    """
    Channel Projection Layer
    """
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_features: int = 1,
                 individual: bool = False):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(input_len, output_len) for _ in range(num_features)
        ]) if individual else nn.Linear(input_len, output_len)
        self.individual = individual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            x_out = []
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))
            x = torch.stack(x_out, dim=-1)
        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


class FactorizedChannelMixing(nn.Module):

    """
    Factorized Channel Mixing Layer
    """

    def __init__(self, input_dim: int, factorized_dim: int) :
        super().__init__()
        self.channel_mixing = MLPLayer(input_dim, factorized_dim, hidden_act="gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixing(x)


class MixerLayer(nn.Module):
    """
    Mixer Layer of MTSMixer
    """
    def __init__(self, config: MTSMixerConfig):
        super().__init__()
        self.temporal_mixing = FactorizedTemporalMixing(
            config.input_len, config.temporal_hidden_size, config.down_sampling
            ) if config.fac_T else MLPLayer(config.input_len, config.temporal_hidden_size)
        self.channel_mixing = FactorizedChannelMixing(
            config.num_features, config.channel_hidden_size) if config.fac_C else None
        self.norm = nn.LayerNorm(config.num_features) if config.use_layer_norm else None

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x) if self.norm else x
        y = self.temporal_mixing(y.transpose(1, 2)).transpose(1, 2)
        if self.channel_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channel_mixing(y)
        return y
