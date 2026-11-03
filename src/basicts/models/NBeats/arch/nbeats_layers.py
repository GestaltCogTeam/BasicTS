from typing import Tuple

import numpy as np
import torch

from basicts.modules import ACT2FN


class NBeatsBlock(torch.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size,
        theta_size: int,
        basis_function: torch.nn.Module,
        layers: int,
        layer_size: int,
        hidden_act: str,
    ):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=input_size, out_features=layer_size)]
            + [
                torch.nn.Linear(in_features=layer_size, out_features=layer_size)
                for _ in range(layers - 1)
            ]
        )
        self.basis_parameters = torch.nn.Linear(
            in_features=layer_size, out_features=theta_size
        )
        self.basis_function = basis_function
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = self.act_fn(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)
