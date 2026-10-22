"""Interpolation forecasting basis from N-HiTS."""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _interp_linear1d(x: torch.Tensor, size: int) -> torch.Tensor:
    """Deterministic 1-D linear resampling (matches ``F.interpolate`` align_corners=False)."""

    length = x.shape[-1]
    if length == size:
        return x
    if size == 1:
        return x[..., :1]
    pos = (torch.arange(size, device=x.device, dtype=x.dtype) + 0.5) * (
        length / size
    ) - 0.5
    pos = pos.clamp(0, length - 1)
    idx0 = pos.floor().to(torch.long)
    idx1 = torch.clamp(idx0 + 1, max=length - 1)
    w1 = pos - idx0.to(x.dtype)
    w0 = 1.0 - w1
    while w0.ndim < x.ndim:
        w0 = w0.unsqueeze(0)
        w1 = w1.unsqueeze(0)
    x0 = x.index_select(-1, idx0)
    x1 = x.index_select(-1, idx1)
    return w0 * x0 + w1 * x1


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
        self,
        backcast_theta: torch.Tensor,
        forecast_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = _interp_linear1d(knots, self.forecast_size)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )  # , align_corners=True)
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]

        return backcast, forecast
