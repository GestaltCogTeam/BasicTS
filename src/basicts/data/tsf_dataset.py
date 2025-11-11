import os
from typing import Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from .base_dataset import BasicTSDataset


class BasicTSForecastingDataset(BasicTSDataset):
    """
    A dataset class for time series forecasting problems.
    
    Attributes:
        dataset_name (str): The name of the dataset.
        input_len (int): The length of the input sequence (number of historical points).
        output_len (int): The length of the output sequence (number of future points to predict).
        mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
        use_timestamps (bool): Flag to determine if timestamps should be used.
        local (bool): Flag to determine if the dataset is local.
        data_file_path (str | None): Path to the file containing the time series data. Default to "datasets/{dataset_name}".
        memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
    """

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            output_len: int,
            mode: Union[BasicTSMode, str],
            use_timestamps: bool = False,
            local: bool = True,
            data_file_path: Union[str, None] = None,
            memmap: bool = False) -> None:
        """
        Initializes the BasicTSForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
            use_timestamps (bool): Flag to determine if timestamps should be used.
            local (bool): Flag to determine if the dataset is local.
            data_file_path (str | None): Path to the file containing the time series data. Default to "datasets/{name}".
            memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
        """
        super().__init__(dataset_name, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        if not local:
            pass # TODO: support download remotely
        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}" # default file path
        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None)
            if use_timestamps:
                self.timestamps = np.load(
                    os.path.join(data_file_path, f"{mode}_timestamps.npy"),
                    mmap_mode="r" if memmap else None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot load dataset from {data_file_path}, Please set a correct local path."\
                                    "If you want to download the dataset, please set the argument `local` to False.") from e
        self.memmap = memmap
        self.use_timestamps = use_timestamps

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing "inputs" and "targets", where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        item = {}
        history_data = self._data[index: index + self.input_len]
        future_data = self._data[index + self.input_len: index + self.input_len + self.output_len]
        item["inputs"] = history_data.copy() if self.memmap else history_data
        item["targets"] = future_data.copy() if self.memmap else future_data
        if self.use_timestamps:
            history_timestamps = self.timestamps[index: index + self.input_len]
            future_timestamps = self.timestamps[index + self.input_len: index + self.input_len + self.output_len]
            item["inputs_timestamps"] = history_timestamps.copy() if self.memmap else history_timestamps
            item["targets_timestamps"] = future_timestamps.copy() if self.memmap else future_timestamps
        return item

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self._data) - self.input_len - self.output_len + 1

    @property
    def data(self) -> np.ndarray:
        return self._data
