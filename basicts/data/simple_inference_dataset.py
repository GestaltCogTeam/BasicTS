import json
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


class TimeSeriesInferenceDataset(BaseDataset):
    """
    A dataset class for time series inference tasks, where the input is a sequence of historical data points
    
    Attributes:
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        description (dict): Metadata about the dataset, such as shape and other properties.
        data (np.ndarray): The loaded time series data array.
        raw_data (str): The raw data path or data list of the dataset.
        last_datetime (pd.Timestamp): The last datetime in the dataset. Used to generate time features of future data.
    """

    # pylint: disable=unused-argument
    def __init__(self, dataset_name:str, dataset: Union[str, list], input_len: int, output_len: int,
                 logger: logging.Logger = None, **kwargs) -> None:
        """
        Initializes the TimeSeriesInferenceDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset. If dataset_name is None, the dataset is expected to be passed directly.
            dataset(str or array): The data path of the dataset or data itself.
            input_len(str): The length of the input sequence (number of historical points).
            output_len(str): The length of the output sequence (number of future points to predict).
            logger (logging.Logger): logger.
        """
        train_val_test_ratio: List[float] = []
        mode: str = 'inference'
        overlap = False
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.description = {}
        if dataset_name:
            self.description_file_path = f'datasets/{dataset_name}/desc.json'
            self.description = self._load_description()

        self.last_datetime:pd.Timestamp = pd.Timestamp.now()
        self._raw_data = dataset
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
        Loads the time series data from a file or list and processes it according to the dataset description.
        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        if isinstance(self._raw_data, str):
            df = pd.read_csv(self._raw_data, header=None)
        else:
            df = pd.DataFrame(self._raw_data)

        df_index = pd.to_datetime(df[0].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
        df = df[df.columns[1:]]
        df.index = pd.Index(df_index)
        df = df.astype('float32')
        self.last_datetime = df.index[-1]

        data = np.expand_dims(df.values, axis=-1)
        data = data[..., [0]]

        # if description is not provided, we assume the data is already in the correct shape.
        if not self.dataset_name:
            # calc frequency form df
            freq = int((df.index[1] - df.index[0]).total_seconds() / 60)  # convert to minutes
            if freq <= 0:
                raise ValueError('Frequency must be a positive number.')
            self.description = {
                'shape': data.shape,
                'frequency (minutes)': freq,
            }

        data_with_features = self._add_temporal_features(data, df)

        data_set_shape = self.description['shape']
        _, n, c = data_with_features.shape
        if data_set_shape[1] != n or data_set_shape[2] != c:
            raise ValueError(f'Error loading data. Shape mismatch: expected {data_set_shape[1:]}, got {[n,c]}.')

        return data_with_features

    def _add_temporal_features(self, data, df) -> np.ndarray:
        '''
        Add time of day and day of week as features to the data.

        Args:
            data (np.ndarray): The data array.
            df (pd.DataFrame): The dataframe containing the datetime index.
        
        Returns:
            np.ndarray: The data array with added time of day and day of week features.
        '''

        _, n, _ = data.shape
        feature_list = [data]

        # numerical time_of_day
        tod = (df.index.hour*60 + df.index.minute) / (24*60)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

        # numerical day_of_week
        dow = df.index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

        # numerical day_of_month
        dom = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

        data_with_features = np.concatenate(feature_list, axis=-1).astype('float32')  # L x N x C

        # Remove extra features
        data_set_shape = self.description['shape']
        data_with_features = data_with_features[..., range(data_set_shape[2])]

        return data_with_features

    def append_data(self, new_data: np.ndarray) -> None:
        """
        Append new data to the existing data

        Args:
            new_data (np.ndarray): The new data to append to the existing data.
        """

        freq = self.description['frequency (minutes)']
        l, _, _ = new_data.shape

        data_with_features, datetime_list = self._gen_datetime_list(new_data, self.last_datetime, freq, l)
        self.last_datetime = datetime_list[-1]

        self.data = np.concatenate([self.data, data_with_features], axis=0)

    def _gen_datetime_list(self, new_data: np.ndarray, start_datetime: pd.Timestamp, freq: int, num_steps: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Generate a list of datetime objects based on the start datetime, frequency, and number of steps.

        Args:
            start_datetime (pd.Timestamp): The starting datetime for the sequence.
            freq (int): The frequency of the data in minutes.
            num_steps (int): The number of steps in the sequence.

        Returns:
            List[pd.Timestamp]: A list of datetime objects corresponding to the sequence.
        """
        datetime_list = [start_datetime]
        for _ in range(num_steps):
            datetime_list.append(datetime_list[-1] + pd.Timedelta(minutes=freq))
        new_index = pd.Index(datetime_list[1:])
        new_df = pd.DataFrame()
        new_df.index = new_index
        data_with_features = self._add_temporal_features(new_data, new_df)

        return data_with_features, datetime_list

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset, considering both the input and output lengths.
        For inference, the input data is the last 'input_len' points in the dataset, and the output data is the next 'output_len' points.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data[-self.input_len:]

        freq = self.description['frequency (minutes)']
        _, n, _ = history_data.shape
        future_data = np.zeros((self.output_len, n, 1))

        data_with_features, _ = self._gen_datetime_list(future_data, self.last_datetime, freq, self.output_len)
        return {'inputs': history_data, 'target': data_with_features}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset.
        For inference, there is only one valid sample, as the input data is the last 'input_len' points in the dataset.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return 1
