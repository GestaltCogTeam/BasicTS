from typing import Union

import numpy as np
from torch.utils.data import Dataset

from basicts.utils.constants import BasicTSMode


class BasicTSDataset(Dataset):
    """
    The base dataset class in BasicTS.
    
    Attributes:
        dataset_name (str): The name of the dataset.
        mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
        memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
    """

    def __init__(self, dataset_name: str, mode: Union[BasicTSMode, str], memmap: bool = False):
        """
        Initializes the BasicTSDataset with the specified dataset name, mode, and memory mapping option.

        Args:
            dataset_name (str): The name of the dataset.
            mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
            memmap (bool, optional): Flag to determine if the dataset should be loaded using memory mapping. Defaults to False.
        """
        self.dataset_name = dataset_name
        self.mode = mode
        self.memmap = memmap

    @property
    def data(self) -> np.ndarray:
        raise NotImplementedError()
