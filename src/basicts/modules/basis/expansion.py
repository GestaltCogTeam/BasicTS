# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

# Modified from:


"""Neural forecasting basis expansions from the N-BEATS implementation."""

import numpy as np
import torch
from torch import nn


class GenericBasis(torch.nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class TrendBasis(torch.nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # degree of polynomial with constant term
        self.backcast_time = torch.nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=float) / backcast_size, i
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.forecast_time = torch.nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=float) / forecast_size, i
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum(
            "bp,pt->bt", theta[:, self.polynomial_size :], self.backcast_time
        )
        forecast = torch.einsum(
            "bp,pt->bt", theta[:, : self.polynomial_size], self.forecast_time
        )
        return backcast, forecast


class SeasonalityBasis(torch.nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )
        self.backcast_cos_template = torch.nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
            requires_grad=False,
        )
        self.backcast_sin_template = torch.nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
            requires_grad=False,
        )
        self.forecast_cos_template = torch.nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
            requires_grad=False,
        )
        self.forecast_sin_template = torch.nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic],
            self.backcast_cos_template,
        )
        backcast_harmonics_sin = torch.einsum(
            "bp,pt->bt", theta[:, 3 * params_per_harmonic :], self.backcast_sin_template
        )
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum(
            "bp,pt->bt", theta[:, :params_per_harmonic], self.forecast_cos_template
        )
        forecast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic : 2 * params_per_harmonic],
            self.forecast_sin_template,
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
