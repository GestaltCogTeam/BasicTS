"""RevIN with explicit statistics conventions for migrated forecasting models.

The upstream RevIN stays unchanged. These options preserve the sample variance
and statistics gradients used by TQNet, BiMamba and S-Mamba.
"""

import torch

from basicts.modules.norm import RevIN


class ForecastRevIN(RevIN):
    """
    Paper: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
    Link: https://openreview.net/forum?id=cGDAkQo1C0p
    Ref Official Code: https://github.com/ts-kim/RevIN
    Venue: ICLR 2022
    Task: Input Normalization for Time Series Forecasting

    Note:
        Optional statistics conventions preserve migrated backbone behavior.
    """

    def __init__(
        self,
        num_features=None,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        unbiased=False,
        detach_mean=True,
        detach_stdev=True,
        center_before_variance=False,
    ):
        super().__init__(num_features, eps, affine, subtract_last)
        self.unbiased = unbiased
        self.detach_mean = detach_mean
        self.detach_stdev = detach_stdev
        self.center_before_variance = center_before_variance

    def _get_statistics(self, x):
        dims = tuple(range(1, x.ndim - 1))
        mean = x.mean(dim=dims, keepdim=True)
        self.mean = mean.detach() if self.detach_mean else mean
        if self.subtract_last:
            self.last = x[:, -1:, :]
        values = x - self.mean if self.center_before_variance else x
        stdev = torch.sqrt(
            values.var(dim=dims, keepdim=True, unbiased=self.unbiased) + self.eps
        )
        self.stdev = stdev.detach() if self.detach_stdev else stdev
