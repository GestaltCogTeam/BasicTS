import torch
import torch.nn as nn


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

#
# class RevIN_loc(nn.Module):
#     def __init__(self, num_features: int, eps=1e-5, affine=True):
#         """
#         :param num_features: the number of features or channels
#         :param eps: a value added for numerical stability
#         :param affine: if True, RevIN has learnable affine parameters
#         """
#         super(RevIN_loc, self).__init__()
#         self.num_features_l = num_features
#         self.eps_l = eps
#         self.affine_l = affine
#         if self.affine_l:
#             self._init_params_l()
#
#     def forward(self, x, mode:str):
#         if mode == 'norm':
#             self._get_statistics_l(x)
#             x = self._normalize_l(x)
#         elif mode == 'denorm':
#             x = self._denormalize_l(x)
#         else: raise NotImplementedError
#         return x
#
#     def _init_params_l(self):
#         # initialize RevIN params: (C,)
#         self.affine_weight_l = nn.Parameter(torch.ones(self.num_features_l))
#         self.affine_bias_l = nn.Parameter(torch.zeros(self.num_features_l))
#
#     def _get_statistics_l(self, x):
#         dim2reduce_l = tuple(range(1, x.ndim-1))
#         self.mean_l = torch.mean(x, dim=dim2reduce_l, keepdim=True).detach()
#         self.stdev_l = torch.sqrt(torch.var(x, dim=dim2reduce_l, keepdim=True, unbiased=False) + self.eps_l).detach()
#
#     def _normalize_l(self, x):
#         x = x - self.mean_l
#         x = x / self.stdev_l
#         if self.affine_l:
#             x = x * self.affine_weight_l
#             x = x + self.affine_bias_l
#         return x
#
#     def _denormalize_l(self, x):
#         if self.affine_l:
#             x = x - self.affine_bias_l
#             x = x / (self.affine_weight_l + self.eps_l*self.eps_l)
#         x = x * self.stdev_l
#         x = x + self.mean_l
#         return x
#
