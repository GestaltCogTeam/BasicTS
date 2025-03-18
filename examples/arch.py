# pylint: disable=unused-argument
from typing import Any, List, Optional
import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl

from basicts.model import BasicTimeSeriesForecastingModule
from basicts.scaler import BaseScaler


class MultiLayerPerceptron(BasicTimeSeriesForecastingModule):
    """
    A simple Multi-Layer Perceptron (MLP) model with two fully connected layers.

    This model is designed to take historical time series data as input and produce future predictions.
    It consists of two linear layers with a ReLU activation in between.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer, which maps the input history sequence to a hidden dimension.
        fc2 (nn.Linear): The second fully connected layer, which maps the hidden dimension to the prediction sequence.
        act (nn.ReLU): The ReLU activation function applied between the two layers.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        history_len: int,
        horizon_len: int,
        hidden_dim: int,
        metrics: Optional[List[str]] = None,
        forward_features: Optional[List[int]] = None,
        target_features: Optional[List[int]] = None,
        target_time_series: Optional[List[int]] = None,
        scaler: Any = None,
        null_val: Any = np.nan,
    ):
        """
        Initialize the MultiLayerPerceptron model.

        Args:
            history_seq_len (int): The length of the input history sequence.
            prediction_seq_len (int): The length of the output prediction sequence.
            hidden_dim (int): The number of units in the hidden layer.

        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            history_len=history_len,
            horizon_len=horizon_len,
            metrics=metrics,
            forward_features=forward_features,
            target_features=target_features,
            target_time_series=target_time_series,
            scaler=scaler,
            null_val=null_val,
        )
        self.fc1 = nn.Linear(history_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, horizon_len)
        self.act = nn.ReLU()

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            history_data (torch.Tensor): A tensor containing historical data, typically of shape `[B, L, N, C]`.
            future_data (torch.Tensor): A tensor containing future data, typically of shape `[B, L, N, C]`.

        Returns:
            torch.Tensor: The output prediction tensor, typically of shape `[B, L, N, C]`.
        """

        history_data = history_data[..., 0].transpose(1, 2)  # [B, L, N, C] -> [B, N, L]

        # [B, N, L] --h=act(fc1(x))--> [B, N, D] --fc2(h)--> [B, N, L] -> [B, L, N]
        prediction = self.fc2(self.act(self.fc1(history_data))).transpose(1, 2)

        # [B, L, N] -> [B, L, N, 1]
        return prediction.unsqueeze(-1)
