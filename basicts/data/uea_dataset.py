import logging
import os
from typing import List

from .simple_tsc_dataset import TimeSeriesClassificationDataset


class UEADataset(TimeSeriesClassificationDataset):
    """
    A UEA dataset class for time series classification problems, handling the loading, parsing, and partitioning
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

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, memmap: bool = False,
                 logger: logging.Logger = None) -> None:
        """
        Initializes the UEADataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        if mode == 'valid':
            mode = 'test'
        dataset_name = os.path.join('UEA', dataset_name)
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap, logger)
