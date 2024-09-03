import json

import torch
import numpy as np

from .base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """
    MinMaxScaler performs min-max normalization on the dataset, scaling the data to a specified range 
    (typically [0, 1] or [-1, 1]).

    Attributes:
        min (np.ndarray): The minimum values of the training data used for normalization. 
            If `norm_each_channel` is True, this is an array of minimum values, one for each channel. Otherwise, it's a single scalar.
        max (np.ndarray): The maximum values of the training data used for normalization. 
            If `norm_each_channel` is True, this is an array of maximum values, one for each channel. Otherwise, it's a single scalar.
        target_channel (int): The specific channel (feature) to which normalization is applied. 
            By default, it is set to 0, indicating the first channel.
    """

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool = True, rescale: bool = True):
        """
        Initialize the MinMaxScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the minimum and maximum values from the training data, which are then used 
        to normalize the data during the `transform` operation.

        Args:
            dataset_name (str): The name of the dataset used to load the data.
            train_ratio (float): The ratio of the dataset to be used for training. The scaler is fitted on this portion of the data.
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the min and max values are computed for each channel independently. Defaults to True.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. 
                This flag is included for consistency with the base class but is typically True in min-max scaling.
        """

        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = 0  # assuming normalization on the first channel

        # load dataset description and data
        description_file_path = f'datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))

        # split data into training set based on the train_ratio
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()

        # compute minimum and maximum values for normalization
        if norm_each_channel:
            self.min = np.min(train_data, axis=0, keepdims=True)
            self.max = np.max(train_data, axis=0, keepdims=True)
        else:
            self.min = np.min(train_data)
            self.max = np.max(train_data)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max normalization to the input data.

        This method normalizes the input data using the minimum and maximum values computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - self.min) / (self.max - self.min)
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the min-max normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the minimum and maximum 
        values computed from the training data. This is useful for interpreting model outputs or for further analysis 
        in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        input_data[..., self.target_channel] = input_data[..., self.target_channel] * (self.max - self.min) + self.min
        return input_data
