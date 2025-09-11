import inspect
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from basicts.utils import BasicTSMode


class BasicTSDataset(Dataset, ABC):
    """
    An abstract base class for creating datasets for time series analysis in PyTorch.

    This class provides a structured template for defining custom datasets by specifying methods
    to load data and descriptions, and to access individual samples. It is designed to be subclassed
    with specific implementations for different types of time series data.

    Attributes:
        name (str): The name of the dataset which is used for identifying the dataset uniquely.
        data_file_path (str): The file path of the dataset.
        memmap (bool): Whether to use memory-mapped file access for loading data.
    """

    _instances: Dict[BasicTSMode, "BasicTSDataset"] = {}

    def __init__(self, name: str, data_file_path: str, memmap: bool, **kwargs):
        super().__init__()
        self.name: str = name
        self.data_file_path: str = data_file_path
        self.memmap: bool = memmap
        self.mode: BasicTSMode = None
        self.data: Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]] = None
        self._kwargs = kwargs

    def __call__(self, mode: BasicTSMode) -> "BasicTSDataset":
        """
        Creating a BasicTSDataset instance, it can be called as a function to set mode to \
        MODE.TRAIN, MODE.VALID, or MODE.TEST.
        """

        if mode not in BasicTSDataset._instances:
            sig = inspect.signature(self.__class__.__init__)
            params = sig.parameters

            kwargs = {"name": self.name, "memmap": self.memmap, **self._kwargs}

            missing_args = [
                param.name
                for param in params.values()
                if (
                    param.default == param.empty
                    and param.name not in kwargs
                    and param.name not in ("self", "args", "kwargs")
                )
            ]
            if missing_args:
                raise ValueError(f"Missing required arguments: {missing_args}")
            new_instance = self.__class__(**kwargs)
            new_instance.mode = mode
            new_instance.data = new_instance.load_data()
            BasicTSDataset._instances[mode] = new_instance

        return BasicTSDataset._instances[mode]

    @abstractmethod
    def load_data(self) -> Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]:
        """
        Load the dataset and organize it based on the specified mode.

        This method should be implemented by subclasses to load actual time series data into an array,
        handling any necessary preprocessing and partitioning according to the specified `mode`.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]: The loaded dataset, which can be a single \
            array, a tuple of arrays, or a dictionary of arrays, depending on the implementation.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        """

        pass

    @abstractmethod
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

    @abstractmethod
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
