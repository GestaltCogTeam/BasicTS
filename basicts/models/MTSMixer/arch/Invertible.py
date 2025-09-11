import torch
import torch.nn as nn

class ModifiedLayerNorm(nn.Module):
    """
    Modified Layer Normalization normalizes vectors along channel dimension and temporal dimensions.
    Input: tensor in shape [B, L, D]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        # The shape of learnable affine parameters is also [num_channels, ], keeping the same as vanilla Layer Normalization.
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        x = x.transpose(1, 2)
        u = x.mean([1, 2], keepdim=True) # Mean along channel and spatial dimension.
        s = (x - u).pow(2).mean([1, 2], keepdim=True) # Variance along channel and spatial dimensions.
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)

        return x.transpose(1, 2)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        
        elif mode == 'denorm':
            x = self._denormalize(x)
        
        else: raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        
        return x


class InvDiff(nn.Module):
    def __init__(self, num_features: int):
        super(InvDiff, self).__init__()

        self.num_features = num_features
        self.pivot = None

    def forward(self, x, mode):
        if mode == 'diff':
            self.pivot = x[:, -1]
            x = torch.diff(x, dim=1)

            return x
        
        elif mode == 'restore':
            y = torch.zeros_like(x)
            y[:, 0] = x[:, 0] + self.pivot
            for idx in range(y.shape[1]-1):
                y[:, idx] = x[:, idx] + y[:, idx-1]
            
            return y
        
        else: raise NotImplementedError
