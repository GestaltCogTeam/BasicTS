import torch
from torch import nn
from argparse import Namespace
from .layer import series_decomp
from .model import Model
from basicts.utils import data_transformation_4_xformer

class ModernTCN(nn.Module):
    """
    Paper: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
    Official Code: https://github.com/luodhhh/ModernTCN
    Link: https://openreview.net/forum?id=vpJMJerXHU
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(ModernTCN, self).__init__()
        configs = Namespace(**model_args)
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len
        self.label_len = configs.label_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride


        # decomp
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = Model(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
            self.model_trend = Model(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
        else:
            self.model = Model(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
        

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        x = x_enc
        te = x_mark_enc
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
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

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=self.label_len)
        #print(x_mark_enc.shape, x_mark_dec.shape)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc)
        return prediction.unsqueeze(-1)