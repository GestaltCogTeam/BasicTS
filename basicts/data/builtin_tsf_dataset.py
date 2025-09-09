import json
import logging

import numpy as np

from basicts.utils import BasicTSMode 
from .tsf_dataset import BasicTSForecastingDataset


class BuiltinTSForecastingDataset(BasicTSForecastingDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
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

    def __init__(self, name: str, input_len: int, output_len: int, null_val: float = np.nan,
                 memmap: bool = False, overlap: bool = False, logger: logging.Logger = None) -> None:
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

        super().__init__(name, input_len, output_len, null_val, memmap, overlap, logger)
        self.data_file_path = f'datasets/{name}/data.dat'
        self.description_file_path = f'datasets/{name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """

        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        self.val_len = int(total_len * self.description['regular_settings']['TRAIN_VAL_TEST_RATIO'][1])
        self.test_len = int(total_len * self.description['regular_settings']['TRAIN_VAL_TEST_RATIO'][2])
        self.train_len = total_len - self.val_len - self.test_len

        # Automatically configure the overlap parameter
        # minimal_len = self.input_len + self.output_len
        # if minimal_len > {MODE.TRAIN: self.train_len, MODE.VALID: self.val_len, MODE.TEST: self.test_len}[self.mode]:
        #     self.overlap = True  # Enable overlap when the train, validation, or test set is too short
        #     current_frame = inspect.currentframe()
        #     file_name = inspect.getfile(current_frame)
        #     line_number = current_frame.f_lineno - 7
        #     if self.logger is not None:
        #         self.logger.info(f'{self.mode} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
        #     else:
        #         print(f'{self.mode} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

        if not self.memmap:
            data = data.copy()
        return data

    def load_data(self) -> np.ndarray:
        """
        Load the dataset and organize it based on the specified mode.

        This method should be implemented by subclasses to load actual time series data into an array,
        handling any necessary preprocessing and partitioning according to the specified `mode`.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]: The loaded dataset, which can be a single \
            array, a tuple of arrays, or a dictionary of arrays, depending on the implementation.
        """

        if self.mode is None:
            raise ValueError(f'Mode is None. The mode should be setted by using `{self.__class__.__name__}(mode)`.')

        if self.mode == BasicTSMode.TRAIN:
            offset = self.output_len if self.overlap else 0
            return self.data[:self.train_len + offset]
        elif self.mode == BasicTSMode.VAL:
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return self.data[self.train_len - offset_left : self.train_len + self.val_len + offset_right]
        else:  # self.mode == MODE.TEST
            offset = self.input_len - 1 if self.overlap else 0
            return self.data[self.train_len + self.val_len - offset:]
