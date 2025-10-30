from typing import Union

import numpy as np
import torch

from .base_scaler import BasicTSScaler


class ZScoreScaler(BasicTSScaler):
    """
    ZScoreScaler performs Z-score normalization on the dataset, transforming the data to have a mean of zero 
    and a standard deviation of one. This is commonly used in preprocessing to normalize data, ensuring that 
    each feature contributes equally to the model.

    Attributes:
        mean (np.ndarray): The mean of the training data used for normalization. 
            If `norm_each_channel` is True, this is an array of means, one for each channel. Otherwise, it's a single scalar.
        std (np.ndarray): The standard deviation of the training data used for normalization.
            If `norm_each_channel` is True, this is an array of standard deviations, one for each channel. Otherwise, it's a single scalar.
        target_channel (int): The specific channel (feature) to which normalization is applied.
            By default, it is set to 0, indicating the first channel.
    """

    def __init__(self, norm_each_channel: bool, rescale: bool, stats: dict = None):
        """
        Initialize the ZScoreScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the mean and standard deviation from the training data, which is then used to 
        normalize the data during the `transform` operation.

        Args:
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the mean and standard deviation are computed for each channel independently.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. This flag is included for consistency with 
                the base class but is not directly used in Z-score normalization.
        """

        super().__init__(norm_each_channel, rescale, stats or {})

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Fit the scaler to the training data.

        Args:
            data (torch.Tensor): Training data used to fit the scaler.
        """

        # load from previous stats
        if self.stats:
            return

        # fit from trainining dataset
        if isinstance(data, np.ndarray):
            if self.norm_each_channel:
                mean = np.mean(data, axis=-2, keepdims=True)
                std = np.std(data, axis=-2, keepdims=True)
                std[std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    std = 1.0  # prevent division by zero by setting std to 1 where it's 0
            self.stats['mean'], self.stats['std'] = torch.Tensor(mean), torch.Tensor(std)
        else:
            if self.norm_each_channel:
                self.stats['mean'] = torch.mean(data, dim=-2, keepdim=True)
                self.stats['std'] = torch.std(data, dim=-2, keepdim=True)
                self.stats['std'][self.stats['std'] == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
            else:
                self.stats['mean'] = torch.mean(data)
                self.stats['std'] = torch.std(data)
                if self.stats['std'] == 0:
                    self.stats['std'] = 1.0  # prevent division by zero by setting std to 1 where it's 0

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply Z-score normalization to the input data.

        This method normalizes the input data using the mean and standard deviation computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be normalized. 
                If None, all elements are normalized. Defaults to None.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        normed_data = (input_data - mean) / std
        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)
        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Reverse the Z-score normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the mean and standard deviation 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be transformed back. 
                If None, all elements are transformed back. Defaults to None.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        mean = self.stats['mean'].to(input_data.device)
        std = self.stats['std'].to(input_data.device)
        denormed_data = input_data * std + mean
        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)
        return denormed_data
