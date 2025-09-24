from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class BasicTSScaler:
    """
    BasicTSScaler is an abstract class for data scaling and normalization methods.

    Attributes:
        dataset_name (str): The name of the dataset, used to load the data.
        train_ratio (float): Ratio of the data to be used for training, for fitting the scaler.
        norm_each_channel (bool): Flag indicating whether to normalize each channel separately.
        rescale (bool): Flag indicating whether to apply rescaling.
    """

    norm_each_channel: bool
    rescale: bool

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Fit the scaler to the training data.

        Args:
            data (torch.Tensor): Training data used to fit the scaler.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the scaling transformation to the input data.

        Args:
            input_data (torch.Tensor): Input data to be transformed.

        Returns:
            torch.Tensor: Scaled data.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse scaling transformation to the input data.

        Args:
            input_data (torch.Tensor): Input data to be transformed back.

        Returns:
            torch.Tensor: Original scale data.
        """

        raise NotImplementedError("Subclasses should implement this method.")
