import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from contextlib import contextmanager

def svd_denoise(x, cut):
    x_ = x.clone().detach()
    U, S, V = torch.linalg.svd(x_, full_matrices=False)
    S[:, cut:] = 0

    return U @ torch.diag(S[0, :]) @ V

@contextmanager
def null_context():
    yield

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class NMF(nn.Module):
    def __init__(self, dim, n, ratio=8, K=6, eps=2e-8):
        super().__init__()
        r = dim // ratio

        D = torch.zeros(dim, r).uniform_(0, 1)
        C = torch.zeros(r, n).uniform_(0, 1)

        self.K = K
        self.D = nn.Parameter(D)
        self.C = nn.Parameter(C)

        self.eps = eps

    def forward(self, x):
        b, D, C, eps = x.shape[0], self.D, self.C, self.eps

        # x is made non-negative with relu as proposed in paper
        x = F.relu(x)

        D = repeat(D, 'd r -> b d r', b = b)
        C = repeat(C, 'r n -> b r n', b = b)

        # transpose
        t = lambda tensor: rearrange(tensor, 'b i j -> b j i')

        for k in reversed(range(self.K)):
            # only calculate gradients on the last step, per propose 'One-step Gradient'
            context = null_context if k == 0 else torch.no_grad
            with context():
                C_new = C * ((t(D) @ x) / ((t(D) @ D @ C) + eps))
                D_new = D * ((x @ t(C)) / ((D @ C @ t(C)) + eps))
                C, D = C_new, D_new

        return D @ C