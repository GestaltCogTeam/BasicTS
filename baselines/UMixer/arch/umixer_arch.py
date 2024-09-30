import torch
import torch.nn as nn
import torch.fft
from .Embed import PatchEmbedding
from .RevIN import RevIN


def S_Correction(x, x_pre):
    x_fft = torch.fft.rfft(x,dim=1,norm='ortho')
    x_pre_fft = torch.fft.rfft(x_pre, dim=1, norm='ortho')
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    x_ifft = torch.fft.irfft(x_fft, dim=1) #
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=1)
    x_ifft = torch.clamp(x_ifft,min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft,min=0)
    alpha = torch.sum(x_ifft*x_pre_ifft,dim=1,keepdim=True)/(torch.sum(x_pre_ifft*x_pre_ifft,dim=1,keepdim=True)+0.001)
    #alpha = (x_ifft * x_pre_ifft) / (x_pre_ifft * x_pre_ifft + 0.001)
    return torch.sqrt(alpha)


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class channelMix_CI_pat(nn.Module):
    def __init__(self, **model_args):
        super(channelMix_CI_pat, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.stride = model_args['stride']
        self.patch_len = model_args['patch_len']
        self.d_model = model_args['d_model']
        self.dropout = model_args["dropout"]
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)

        self.conv1 = nn.ModuleList(nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        self.conv2 = nn.ModuleList(nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.d_model

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device='cuda:0')
        for i in range(self.channels):
            o[:, :, i] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, i]))))
        res = o + x
        res = self.norm(res)
        return res


class tempolMix_CI_pat(nn.Module):
    def __init__(self, **model_args):
        super(tempolMix_CI_pat, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.stride = model_args['stride']
        self.patch_len = model_args['patch_len']
        self.d_model = model_args['d_model']
        self.dropout = model_args["dropout"]
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)

        self.conv1 = nn.ModuleList(nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        self.conv2 = nn.ModuleList(nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.Pnum

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device='cuda:0')
        for i in range(self.channels):
            o[:, i, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, i, :]))))
        res = o + x
        res = self.norm(res)
        return res


class UMixer(nn.Module):
    """
    Paper: U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting
    Official Code: https://github.com/XiangMa-Shaun/U-Mixer
    Link: https://arxiv.org/abs/2401.02236
    """
    def __init__(self,  **model_args):
        super(UMixer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.stride = model_args['stride']
        self.patch_len = model_args['patch_len']
        self.d_model = model_args['d_model']
        self.dropout = model_args["dropout"]
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)

        self.layer = model_args['e_layers']

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len+self.seq_len)
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.enc_in = model_args['enc_in']
        self.dropout = model_args["dropout"]
        self.c_out = model_args['c_out']

        self.mlp_tempmix_md = nn.ModuleList([tempolMix_CI_pat(**model_args)
                       for _ in range(self.e_layers)])
        self.mlp_chanmix_md = nn.ModuleList([channelMix_CI_pat(**model_args)
                       for _ in range(self.e_layers)])
        self.mlp_tempmix_mu = nn.ModuleList([tempolMix_CI_pat(**model_args)
                                            for _ in range(self.e_layers)])
        self.mlp_chanmix_mu = nn.ModuleList([channelMix_CI_pat(**model_args)
                                            for _ in range(self.e_layers)])

        self.mlp_trend_ci = nn.ModuleList(nn.Linear(self.pred_len, self.d_model) for _ in range(self.c_out))
        self.mlp_trend2_ci = nn.ModuleList(nn.Linear(self.d_model, self.pred_len) for _ in range(self.c_out))

        self.revin = RevIN(self.enc_in)
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.dropout)
        self.head = Flatten_Head(self.enc_in, self.d_model * self.Pnum, self.pred_len,
                                 head_dropout=self.dropout)
        self.comb = nn.Linear(self.e_layers, 1)

    def forecast(self, x_input):
        x_ori = x_input.contiguous()
        x_input = self.revin(x_input, 'norm')
        x_input = self.predict_linear(x_input.permute(0, 2, 1))
        x_input, n_vars = self.patch_embedding(x_input)

        x_old, _ = self.patch_embedding(x_ori.permute(0, 2, 1))

        x_all = torch.zeros([x_input.shape[0],x_input.shape[1],x_input.shape[2],self.layer], device='cuda:0')
        for i in range(self.layer):
            x_ud = self.mlp_tempmix_md[i](x_input)
            x_ud = self.mlp_chanmix_md[i](x_ud)
            for i in range(i,-1,-1):
                x_ud = self.mlp_tempmix_mu[i](x_ud)
                x_ud = self.mlp_chanmix_mu[i](x_ud)
            x_all[:,:,:,i] = x_ud
        x_input = self.comb(x_all).squeeze(-1)
        x_input = S_Correction(self.layer_norm(x_old), self.layer_norm(x_input[:, :x_old.shape[1], :])) * x_input
        x_input = torch.reshape(
            x_input, (-1, n_vars, x_input.shape[-2], x_input.shape[-1]))
        x_input = x_input.permute(0, 1, 3, 2)

        x_input = self.head(x_input)
        x_input = x_input.permute(0, 2, 1)
        x_input = self.revin(x_input, 'denorm')

        x = x_input[:,-self.pred_len:,:]

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
        x_input = history_data[:, :, :, 0]

        out = self.forecast(x_input)
        return out.unsqueeze(-1)  # [B, L, C]
