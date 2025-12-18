import torch
from torch import nn

from ..callback.koopa_mask_init import KoopaMaskInitCallback
from ..config.koopa_config import KoopaConfig
from .layers import MLP, FourierFilter, TimeInvKP, TimeVarKP


class Koopa(nn.Module):
    """
    Paper: Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors
    Official Code: https://github.com/thuml/Koopa
    Link: https://arxiv.org/abs/2305.18803
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    """

    _required_callbacks: list[type] = [KoopaMaskInitCallback]

    def __init__(self, config: KoopaConfig):
        super().__init__()
        self.mask_spectrum = None
        self.amps = None
        self.alpha = config.alpha
        self.enc_in = config.enc_in
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.seg_len = config.seg_len
        self.num_blocks = config.num_blocks
        self.dynamic_dim = config.dynamic_dim
        self.hidden_dim = config.hidden_dim
        self.hidden_layers = config.hidden_layers
        self.multistep = config.multistep
        self.disentanglement = FourierFilter(self.mask_spectrum)
        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        # fix: use self.output_len instead of non-existent attribute
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.output_len, activation='relu',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        # separate module lists for time-invariant and time-variant KPs
        self.time_inv_kps = nn.ModuleList([
            TimeInvKP(input_len=self.input_len,
                      pred_len=self.output_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_inv_encoder,
                      decoder=self.time_inv_decoder)
            for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len * self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                                   hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len * self.enc_in, activation='tanh',
                                   hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(enc_in=self.enc_in,
                      input_len=self.input_len,
                      pred_len=self.output_len,
                      seg_len=self.seg_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_var_encoder,
                      decoder=self.time_var_decoder,
                      multistep=self.multistep)
            for _ in range(self.num_blocks)])
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Single-`inputs` forward to match runner API.

        Args:
            inputs (torch.Tensor): history input with shape [B, L, C] or [B, L, C, 1]

        Returns:
            torch.Tensor: prediction tensor with shape [B, output_len, num_features] (may include trailing feature dim)
        """
        history_data = inputs

        if history_data.dim() == 4:
            x_enc = history_data[..., 0]
        elif history_data.dim() == 3:
            x_enc = history_data
        else:
            raise ValueError(f'Unsupported inputs shape: {tuple(history_data.shape)}')

        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        if self.disentanglement is None:
            raise ValueError('Koopa mask_spectrum is not initialized.')

        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = time_inv_output + time_var_output
            else:
                forecast += (time_inv_output + time_var_output)
        res = forecast * std_enc + mean_enc
        if history_data is not None and history_data.dim() == 4 and res.dim() == 3:
            res = res.unsqueeze(-1)
        return res
