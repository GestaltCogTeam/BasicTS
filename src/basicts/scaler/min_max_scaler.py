from typing import Union

import numpy as np
import torch

from .base_scaler import BasicTSScaler


class MinMaxScaler(BasicTSScaler):

    """
    MinMaxScaler performs min-max normalization on the dataset, scaling the data to a specified range 
    (typically [0, 1] or [-1, 1]).
    """

    def __init__(self, norm_each_channel: bool, rescale: bool, stats: dict = None):
        """
        Initialize the MinMaxScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the minimum and maximum values from the training data, which are then used to 
        normalize the data during the `transform` operation.

        Args:
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the minimum and maximum values are computed for each channel independently.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. This flag is included for consistency with 
                the base class but is not directly used in min-max normalization.
            stats (dict, optional): Precomputed statistics (minimum and maximum values) for the scaler. 
                If provided, these values will be used instead of fitting the scaler to the data. Defaults to None.
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
                _min = np.min(data, axis=0, keepdims=True)
                _max_min = np.max(data, axis=-2, keepdims=True) - _min
                _max_min[_max_min == 0] = 1.0
            else:
                _min = np.min(data)
                _max_min = np.max(data) - _min
                if _max_min == 0:
                    _max_min = 1.0
            self.stats['min'], self.stats['max-min'] = torch.Tensor(_min), torch.Tensor(_max_min)
        else:
            if self.norm_each_channel:
                self.stats['min'] = torch.min(data, dim=-2, keepdim=True)
                self.stats['max-min'] = torch.max(data, dim=-2, keepdim=True) - self.stats['min']
                self.stats['max-min'][self.stats['max-min'] == 0] = 1.0
            else:
                self.stats['min'] = torch.min(data)
                self.stats['max-min'] = torch.max(data) - self.stats['min']
                if self.stats['max-min'] == 0:
                    self.stats['max-min'] = 1.0

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply min-max normalization to the input data.

        This method normalizes the input data using the minimum and maximum values computed from the training data. 
        The normalization is applied only to the specified `target_channel`.
        Args:
            input_data (torch.Tensor): The input data to be normalized.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be normalized. 
                If None, all elements are normalized. Defaults to None.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        _min = self.stats['min'].to(input_data.device)
        _max_min = self.stats['max-min'].to(input_data.device)
        normed_data = (input_data - _min) / _max_min
        if mask is not None:
            normed_data = torch.where(mask, normed_data, input_data)
        return normed_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Reverse the min-max normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the minimum and maximum values 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.
            mask (torch.Tensor, optional): A boolean mask indicating which elements of the input_data should be transformed back. 
                If None, all elements are transformed back. Defaults to None.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        _min = self.stats['min'].to(input_data.device)
        _max_min = self.stats['max-min'].to(input_data.device)
        denormed_data = input_data * _max_min + _min
        if mask is not None:
            denormed_data = torch.where(mask, denormed_data, input_data)
        return denormed_data
