import torch
import torch.nn as nn
from .Invertible import RevIN
from .Projection import ChannelProjection
from .decomposition import svd_denoise, NMF

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling) :
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class MTSMixer(nn.Module):
    """
    Paper: MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing
    Official Code: https://github.com/plumprc/MTS-Mixers
    Link: https://arxiv.org/abs/2302.04501
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """
    def __init__(self,  **model_args):
        super(MTSMixer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.d_model = model_args['d_model']
        self.d_ff = model_args['d_ff']
        self.norm = model_args['use_norm']
        self.e_layers = model_args['e_layers']
        self.fac_T = model_args['fac_T']
        self.fac_C = model_args['fac_C']
        self.sampling = model_args['sampling']
        self.individual = model_args['individual']
        self.rev = model_args['rev']
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(self.seq_len, self.enc_in, self.d_model, self.d_ff, self.fac_T, self.fac_C, self.sampling, self.norm)
            for _ in range(self.e_layers)
        ])
        self.norm = nn.LayerNorm(self.enc_in) if self.norm else None
        self.projection = ChannelProjection(self.seq_len, self.pred_len, self.enc_in, self.individual)
        # self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.refine = MLPBlock(configs.pred_len, configs.d_model) if configs.refine else None
        self.rev = RevIN(self.enc_in) if self.rev else None

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        x = history_data[:, :, :, 0]
        x = self.rev(x, 'norm') if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        # x = self.refine(x.transpose(1, 2)).transpose(1, 2) if self.refine else x
        x = self.rev(x, 'denorm') if self.rev else x

        return x.unsqueeze(-1)
