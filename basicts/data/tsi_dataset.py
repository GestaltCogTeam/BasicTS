import logging
from abc import abstractmethod

import numpy as np

from .base_dataset import BasicTSDataset


class BasicTSImputationDataset(BasicTSDataset):
    """
    A dataset class for time series imputation problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    """

    def __init__(self, name: str, data_file_path: str, seq_len: int, memmap: bool = False, overlap: bool = False, logger: logging.Logger = None) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'val', 'test'].
        """
        super().__init__(name=name, data_file_path=data_file_path, seq_len=seq_len, memmap=memmap, overlap=overlap, logger=logger)
        self.seq_len = seq_len
        self.overlap = overlap
        self.logger = logger

    @abstractmethod
    def load_data(self) -> np.ndarray:
        pass

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        inputs = self.data[index:index + self.seq_len]
        if self.memmap:
            inputs = inputs.copy()
        return {'inputs': inputs}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.seq_len + 1
