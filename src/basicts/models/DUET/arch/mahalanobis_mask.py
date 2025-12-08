# pylint: disable=not-callable
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MahalanobisMask(nn.Module):

    """
    Mahalanobis mask module.

    Args:
        input_len (int): Input sequence length.
    """

    def __init__(self, input_len: int):
        super().__init__()
        freq_size = input_len // 2 + 1
        self.A = nn.Parameter(torch.randn(freq_size, freq_size))

    def _calculate_prob_distance(self, x):
        XF = torch.abs(torch.fft.rfft(x, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)
        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()
        p = exp_dist / exp_max
        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)
        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99
        return p

    def _bernoulli_gumbel_rsample(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        b, c, d = dist_matrix.shape
        flatten_matrix = rearrange(dist_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = F.gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d)
        return resample_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._calculate_prob_distance(x)
        sample = self._bernoulli_gumbel_rsample(p)
        mask = sample.unsqueeze(1).to(torch.bool)
        mask = torch.where(mask, 0, -torch.inf)
        return mask
