import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss



def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(inputs.device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            new = new.to(inputs.device)
            c = F.linear(c, self.A.to(inputs.device)) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                                 torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FiLM(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2205.08897
    """
    def __init__(self, **model_args):
        super(FiLM, self).__init__()

        self.seq_len = model_args['seq_len']
        self.pred_len = self.seq_len if model_args['pred_len'] == 0 else model_args['pred_len']

        self.output_attention = model_args['output_attention']
        self.layers = model_args['e_layers']
        self.enc_in = model_args['enc_in']
        self.e_layers = model_args['e_layers']
        # b, s, f means b, f
        self.affine_weight = nn.Parameter(torch.ones(1, 1, model_args['enc_in']))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, model_args['enc_in']))

        self.multiscale = model_args['multiscale'] # 1 2 4
        self.window_size = model_args['window_size'] # 256
        self.ratio = model_args['ratio']
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,
                                                         seq_len=min(self.pred_len, self.seq_len),
                                                         ratio=self.ratio) for n in
                                          self.window_size for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # Normalization from Non-stationary Transformer
        x_enc = history_data[:, :, :, 0]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec.unsqueeze(-1)

