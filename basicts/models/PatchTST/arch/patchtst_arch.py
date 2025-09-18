# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .patchtst_backbone import PatchTST_backbone, series_decomp

class PatchTST(nn.Module):
    """
    Paper: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    Link: https://arxiv.org/abs/2211.14730
    Official Code: https://github.com/yuqinie98/PatchTST
    Venue: ICLR 2023
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, enc_in, seq_len, pred_len, e_layers, n_heads, d_model, d_ff,
                 dropout, fc_dropout, head_dropout, patch_len, stride, individual,
                 padding_patch, revin, affine, subtract_last, decomposition, kernel_size,
                 max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 norm:str='BatchNorm', attn_dropout:float=0., act:str="gelu",
                 key_padding_mask:bool='auto',padding_var:Optional[int]=None,
                 attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros',
                 learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten',
                 verbose:bool=False, **kwargs):
    
        super().__init__()

        # load parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len
    
        n_layers = e_layers
        n_heads = n_heads
        d_model = d_model
        d_ff = d_ff
        dropout = dropout
        fc_dropout = fc_dropout
        head_dropout = head_dropout
    
        individual = individual

        patch_len = patch_len
        stride = stride
        padding_patch = padding_patch
    
        revin = revin
        affine = affine
        subtract_last = subtract_last
    
        decomposition = decomposition
        kernel_size = kernel_size
    
    
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of PatchTST.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        assert history_data.shape[-1] == 1      # only use the target feature
        x = history_data[..., 0]     # B, L, N
    
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x.unsqueeze(-1)
