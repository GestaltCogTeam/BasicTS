import os
import random

import torch
from torch.utils.data import Dataset

from ...utils import load_pkl


class M4ForecastingDataset(Dataset):
    """
    BasicTS tries its best to follow the commonly-used processing approaches of M4 dataset, while also providing more flexible interfaces.
    M4 dataset differs from general MTS datasets in the following aspects:
        - M4 dataset is a univariate time series dataset, which does not sample in a synchronized manner.
            In the state-of-the-art M4 prediction solutions, NBeats [1], the authors first sample ids of the time series and then randomly sample the time series data for each time series.
        - Padding and masking are used to make training more flexible and robust.
        - There is no normalization in M4 dataset.
        - There is no validation dataset in M4 dataset.
        - The test data is the last sample of each time series.
        - The future sequence length is fixed for different subsets.

    Reference:
        [1] N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
        [2] https://github.com/ServiceNow/N-BEATS/blob/master/common/sampler.py
    """

    def __init__(self, data_file_path: str, index_file_path: str, mask_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, mask_file_path)
        # read raw data (normalized)
        self.data = load_pkl(data_file_path)[mode] # padded data: List[List]
        self.mask = load_pkl(mask_file_path)[mode] # padded mask: List[List]
        # read index
        self.index = load_pkl(index_file_path)[mode] # train/test index of each time series: List[List]

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, mask_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if not os.path.isfile(mask_file_path):
            raise FileNotFoundError("BasicTS can not find mask file {0}".format(mask_file_path))

    def __getitem__(self, ts_id: int) -> tuple:
        """Get a sample.

        Args:
            ts_id (int): the iteration index, i.e., the time series id (not the self.index).

        Returns:
            tuple: future_data, history_data, future_mask, history_mask, where the shape of data is L x C and mask is L.
        """

        ts_idxs = list(self.index[ts_id])
        # random select a time series sample
        idx = ts_idxs[random.randint(0, len(ts_idxs)-1)]

        history_data = torch.Tensor(self.data[ts_id][idx[0]:idx[1]]).unsqueeze(1).float()
        future_data = torch.Tensor(self.data[ts_id][idx[1]:idx[2]]).unsqueeze(1).float()
        history_mask = torch.Tensor(self.mask[ts_id][idx[0]:idx[1]]).unsqueeze(1).float()
        future_mask = torch.Tensor(self.mask[ts_id][idx[1]:idx[2]]).unsqueeze(1).float()

        return future_data, history_data, future_mask, history_mask

    def __len__(self):
        """Dataset length (=number of time series)

        Returns:
            int: dataset length
        """

        return len(self.data)
