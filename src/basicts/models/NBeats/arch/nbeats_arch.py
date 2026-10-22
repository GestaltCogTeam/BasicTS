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

"""
N-BEATS Model.
"""

import numpy as np
import torch
from torch import nn

from basicts.modules.basis import GenericBasis, SeasonalityBasis, TrendBasis
from basicts.modules.norm import RevIN

from ..config.nbeats_config import NBeatsConfig
from .nbeats_layers import NBeatsBlock


class NBeats(nn.Module):
    """
    Paper: N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
    Link: https://arxiv.org/abs/1905.10437
    Ref Official Code: https://github.com/ServiceNow/N-BEATS
    Venue: ICLR 2020
    Task: Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-06, "affine": False}

    def __init__(self, config: NBeatsConfig):
        super().__init__()

        self.config = config
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.model_type = config.model_type  # "generic" or "interpretable"

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=False)

        if self.model_type == "generic":
            self.blocks = torch.nn.ModuleList(
                [
                    NBeatsBlock(
                        input_size=self.input_len,
                        theta_size=self.input_len + self.output_len,
                        basis_function=GenericBasis(
                            backcast_size=self.input_len, forecast_size=self.output_len
                        ),
                        layers=config.num_layers,
                        layer_size=config.hidden_size,
                        hidden_act=config.hidden_act,
                    )
                    for _ in range(config.stacks)
                ]
            )

        else:
            # trend block
            trend_block = NBeatsBlock(
                input_size=self.input_len,
                theta_size=2 * (config.degree_of_polynomial + 1),
                basis_function=TrendBasis(
                    degree_of_polynomial=config.degree_of_polynomial,
                    backcast_size=self.input_len,
                    forecast_size=self.output_len,
                ),
                layers=config.num_layers,
                layer_size=config.hidden_size,
                hidden_act=config.hidden_act,
            )

            # seasonality block
            seasonality_block = NBeatsBlock(
                input_size=self.input_len,
                theta_size=4
                * int(
                    np.ceil(config.num_of_harmonics / 2 * self.output_len)
                    - (config.num_of_harmonics - 1)
                ),
                basis_function=SeasonalityBasis(
                    harmonics=config.num_of_harmonics,
                    backcast_size=self.input_len,
                    forecast_size=self.output_len,
                ),
                layers=config.num_layers,
                layer_size=config.hidden_size,
                hidden_act=config.hidden_act,
            )

            self.blocks = torch.nn.ModuleList(
                [trend_block for _ in range(config.stacks)]
                + [seasonality_block for _ in range(config.stacks)]
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            inputs (torch.Tensor): history data with shape [batch_size, input_len, num_features)

        Returns:
            torch.Tensor: prediction with shape [batch_size, output_len, num_features]
        """
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        B, L, N = inputs.shape
        inputs = inputs.transpose(1, 2)  # [B, N, L, 1]
        inputs = inputs.reshape(B * N, L, 1)

        x = inputs.squeeze()

        residuals = x.flip(dims=(1,))
        prediction = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_prediction = block(residuals)
            residuals = residuals - backcast
            prediction = prediction + block_prediction
        prediction = prediction.reshape(B, N, prediction.shape[-1]).transpose(1, 2)

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        return prediction
