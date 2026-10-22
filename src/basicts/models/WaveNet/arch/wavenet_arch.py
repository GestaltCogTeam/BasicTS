import torch
from torch import nn
import torch.nn.functional as F

from ..config.wavenet_config import WaveNetConfig


class WaveNet(nn.Module):
    """
    Paper: WaveNet: A Generative Model for Raw Audio
    Link: https://arxiv.org/abs/1609.03499
    Ref Official Code: Not specified in the source implementation; see Modified from.
    Venue: SSW 2016 (demo); arXiv 2016
    Task: Time Series Forecasting (adaptation of an audio-generation model)

    Modified from:
        https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
        https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py

    Note:
        This implementation adapts the raw-audio WaveNet architecture for forecasting.
        The implementation references above retain the attribution in the ARIES source.
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: WaveNetConfig):
        super().__init__()
        self.output_len = config.output_len
        in_dim = config.in_dim
        out_dim = config.output_len
        residual_channels = config.residual_channels
        dilation_channels = config.dilation_channels
        skip_channels = config.skip_channels
        end_channels = config.end_channels
        kernel_size = config.kernel_size
        blocks = config.blocks
        layers = config.layers

        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.receptive_field = receptive_field

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs: [B, L_in, N]
        Returns:
            [B, L_out, N]
        """
        # Channel-independent conv over each variate: [B, C, N, L]
        x = inputs.unsqueeze(-1).transpose(1, 3).contiguous()
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = skip[:, :, :, -s.size(3) :]
            else:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # [B, L_out, N, T] -> [B, L_out, N]
        return x[..., -1]
