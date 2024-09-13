from typing import List
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class BaseDataset(Dataset):
    """
    An abstract base class for creating datasets for time series forecasting in PyTorch.

    This class provides a structured template for defining custom datasets by specifying methods
    to load data and descriptions, and to access individual samples. It is designed to be subclassed
    with specific implementations for different types of time series data.

    Attributes:
        dataset_name (str): The name of the dataset which is used for identifying the dataset uniquely.
        train_val_test_ratio (List[float]): Ratios for splitting the dataset into training, validation,
            and testing sets respectively. Each value in the list should sum to 1.0.
        mode (str): Operational mode of the dataset. Valid values are "train", "valid", or "test".
        input_len (int): The length of the input sequence, i.e., the number of historical data points used.
        output_len (int): The length of the output sequence, i.e., the number of future data points predicted.
        overlap (bool): Flag to indicate whether the splits between training, validation, and testing can overlap.
            Defaults to True but can be set to False to enforce non-overlapping data in different sets.
    """

    dataset_name: str
    train_val_test_ratio: List[float]
    mode: str
    input_len: int
    output_len: int
    overlap: bool = False

    def _load_description(self) -> dict:
        """
        Abstract method to load a dataset's description from a file or source.

        This method should be implemented by subclasses to load and return the dataset's metadata, 
        such as its shape, range, or other relevant properties, typically from a JSON or similar file.

        Returns:
            dict: A dictionary containing the dataset's metadata.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _load_data(self) -> np.ndarray:
        """
        Abstract method to load the dataset and organize it based on the specified mode.

        This method should be implemented by subclasses to load actual time series data into an array,
        handling any necessary preprocessing and partitioning according to the specified `mode`.

        Returns:
            np.ndarray: The loaded and appropriately split dataset array.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self) -> int:
        """
        Abstract method to get the total number of samples available in the dataset.

        This method should be implemented by subclasses to calculate and return the total number of valid
        samples available for training, validation, or testing based on the configuration and dataset size.

        Returns:
            int: The total number of samples.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, idx: int) -> dict:
        """
        Abstract method to retrieve a single sample from the dataset.

        This method should be implemented by subclasses to access and return a specific sample from the dataset,
        given an index. It should handle the slicing of input and output sequences according to the defined
        `input_len` and `output_len`.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input sequence ('inputs') and output sequence ('target').

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method.")
