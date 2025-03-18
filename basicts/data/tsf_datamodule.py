from typing import List

from basicts.utils import get_regular_settings
from .base_dataset import BaseDataset
import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader


class TimeSeriesForecastingModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class: str,
        dataset_name: str,
        train_val_test_ratio: List[float],
        input_len: int,
        output_len: int,
        overlap: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        prefetch: bool = False,
    ):
        super().__init__()
        dataset_class_packeg, dataset_class_name = dataset_class.rsplit(".", 1)
        self.dataset_class = getattr(
            import_module(dataset_class_packeg), dataset_class_name
        )
        if prefetch:
            # todo: implement DataLoaderX
            # self.dataloader_class = DataLoaderX
            raise NotImplementedError("DataLoaderX is not implemented yet.")
        else:
            self.dataloader_class = DataLoader
        self.dataset_name = dataset_name
        self.train_val_test_ratio = train_val_test_ratio
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.regular_settings = get_regular_settings(dataset_name)


        # self.train_set = TimeSeriesForecastingDataset()

    @property
    def dataset_params(self):
        return {
            "dataset_name": self.dataset_name,
            "train_val_test_ratio": self.train_val_test_ratio,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "overlap": self.overlap,
        }

    def train_dataloader(self):
        """Build train dataset and dataloader.

        Returns:
            train data loader (DataLoader)
        """
        dataset = self.dataset_class(**self.dataset_params, mode="train")
        loader = self.dataloader_class(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        """Build validation dataset and dataloader.

        Returns:
            validation data loader (DataLoader)
        """
        dataset = self.dataset_class(**self.dataset_params, mode="valid")
        loader = self.dataloader_class(
            dataset,
            batch_size=self.batch_size,
            # shuffle=self.shuffle,   # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        """Build test dataset and dataloader.

        Returns:
            test data loader (DataLoader)
        """
        dataset = self.dataset_class(**self.dataset_params, mode="test")
        loader = self.dataloader_class(
            dataset,
            batch_size=self.batch_size,
            # shuffle=self.shuffle, # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
