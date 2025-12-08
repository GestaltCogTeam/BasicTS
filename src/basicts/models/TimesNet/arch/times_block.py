# pylint: disable=not-callable
import torch
import torch.fft
import torch.nn.functional as F
from torch import nn

from ..config.timesnet_config import TimesNetConfig
from .conv_block import InceptionBlockV1


class TimesBlock(nn.Module):

    """"
    TimesBlock in TimesNet.
    """

    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.top_k = config.top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(config.hidden_size, config.intermediate_size,
                               num_kernels=config.num_kernels),
            nn.GELU(),
            InceptionBlockV1(config.intermediate_size, config.hidden_size,
                               num_kernels=config.num_kernels)
        )

    def _fft_for_period(self, x: torch.Tensor):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        freq_list = abs(xf).mean(dim=(0, -1))
        freq_list[0] = 0
        _, top_idx = torch.topk(freq_list, self.top_k)
        top_idx = top_idx.detach().cpu().numpy()
        periods = x.shape[1] // top_idx
        amplitudes = abs(xf).mean(dim=-1)[:, top_idx]
        return periods, amplitudes

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, T, hidden_size = hidden_states.size()
        period_list, period_weight = self._fft_for_period(hidden_states)

        period_states = []
        total_len = self.input_len + self.output_len
        for period in period_list:
            # padding
            if total_len % period != 0:
                padded_len = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    [batch_size, padded_len - total_len, hidden_size]).to(hidden_states.device)
                out = torch.cat([hidden_states, padding], dim=1)
            else:
                padded_len = total_len
                out = hidden_states
            # reshape
            out = out.reshape(
                batch_size, padded_len // period, period, hidden_size).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, hidden_size)
            period_states.append(out[:, :(self.input_len + self.output_len), :])
        period_states = torch.stack(period_states, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, hidden_size, 1)
        period_states = torch.sum(period_states * period_weight, -1)
        # residual connection
        period_states = period_states + hidden_states
        return period_states
