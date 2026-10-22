"""ModernTCN forecasting architecture."""

from ..config.moderntcn_config import ModernTCNConfig
import torch
from torch import nn
from .layer import series_decomp
from .model import Model


class ModernTCN(nn.Module):
    """
    Paper: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
    Link: https://openreview.net/forum?id=vpJMJerXHU
    Ref Official Code: https://github.com/luodhhh/ModernTCN
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: ModernTCNConfig):
        super(ModernTCN, self).__init__()
        self.stem_ratio = config.stem_ratio
        self.downsample_ratio = config.downsample_ratio
        self.ffn_ratio = config.ffn_ratio
        self.num_blocks = config.num_blocks
        self.large_size = config.large_kernel_sizes
        self.small_size = config.small_kernel_sizes
        self.dims = config.dims
        self.dw_dims = config.depthwise_dims
        self.nvars = config.num_features
        self.small_kernel_merged = config.small_kernel_merged
        self.drop_backbone = config.dropout
        self.drop_head = config.head_dropout
        self.use_multi_scale = config.use_multi_scale
        self.revin = config.use_revin
        self.affine = config.affine
        self.subtract_last = config.subtract_last
        self.freq = "h"
        self.seq_len = config.input_len
        self.c_in = (self.nvars,)
        self.individual = config.individual
        self.target_window = config.output_len
        self.label_len = config.input_len // 2
        self.kernel_size = config.moving_avg
        self.patch_size = config.patch_size
        self.patch_stride = config.patch_stride
        self.decomposition = config.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = Model(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )
            self.model_trend = Model(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )
        else:
            self.model = Model(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def forward_xformer(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor
    ) -> torch.Tensor:
        x = x_enc
        te = x_mark_enc
        if self.decomposition:
            (res_init, trend_init) = self.decomp_module(x)
            (res_init, trend_init) = (
                res_init.permute(0, 2, 1),
                trend_init.permute(0, 2, 1),
            )
            if te is not None:
                te = te.permute(0, 2, 1)
            res = self.model_res(res_init, te)
            trend = self.model_trend(trend_init, te)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            x = self.model(x, te)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, inputs):
        marks = inputs.new_empty(inputs.shape[0], inputs.shape[1], 0)
        return self.forward_xformer(inputs.float(), marks)
