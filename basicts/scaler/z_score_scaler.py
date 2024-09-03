import json

import torch
import numpy as np

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):
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

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool):
        """
        Initialize the ZScoreScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the mean and standard deviation from the training data, which is then used to 
        normalize the data during the `transform` operation.

        Args:
            dataset_name (str): The name of the dataset used to load the data.
            train_ratio (float): The ratio of the dataset to be used for training. The scaler is fitted on this portion of the data.
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the mean and standard deviation are computed for each channel independently.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. This flag is included for consistency with 
                the base class but is not directly used in Z-score normalization.
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

        # compute mean and standard deviation
        if norm_each_channel:
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
        else:
            self.mean = np.mean(train_data)
            self.std = np.std(train_data)
            if self.std == 0:
                self.std = 1.0  # prevent division by zero by setting std to 1 where it's 0

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization to the input data.

        This method normalizes the input data using the mean and standard deviation computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - self.mean) / self.std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the Z-score normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the mean and standard deviation 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.

        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        if isinstance(self.mean, np.ndarray):
            self.mean = torch.tensor(self.mean, device=input_data.device)
            self.std = torch.tensor(self.std, device=input_data.device)
        # Clone the input data to prevent in-place modification (which is not allowed in PyTorch)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * self.std + self.mean
        return input_data
