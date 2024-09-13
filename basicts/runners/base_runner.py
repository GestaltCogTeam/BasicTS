import os
import time
from abc import abstractmethod
from typing import Dict, Optional

import torch
import setproctitle
from torch import nn
from torch.utils.data import DataLoader, Dataset

from easytorch import Runner
from easytorch.utils import master_only
from easytorch.core.data_loader import build_data_loader


class BaseRunner(Runner):
    """
    An extended EasyTorch Runner for benchmarking time series models.

    This class provides support for a test data loader and a test process in addition to the standard
    training and validation processes.
    """

    def __init__(self, cfg: Dict) -> None:
        """
        Initialize the BaseRunner.

        Args:
            cfg (Dict): Configuration dictionary containing all relevant settings.
        """

        super().__init__(cfg)

        # validate every `val_interval` epochs if configured
        self.val_interval = cfg.get('VAL', {}).get('INTERVAL', 1)
        # test every `test_interval` epochs if configured
        self.test_interval = cfg.get('TEST', {}).get('INTERVAL', 1)

        # declare data loaders
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # ensure compatibility with higher versions of EasyTorch
        if not hasattr(self, 'to_running_device'):
            from easytorch.device import to_device
            self.to_running_device = to_device

        # set process title
        proctitle_name = f"{cfg['MODEL'].get('NAME')}({cfg.get('DATASET', {}).get('NAME', 'Unknown Dataset')})"
        setproctitle.setproctitle(f'{proctitle_name}@BasicTS')

    def define_model(self, cfg: Dict) -> nn.Module:
        """
        Define the model architecture based on the configuration.

        Args:
            cfg (Dict): Configuration dictionary containing model settings.

        Returns:
            nn.Module: The model architecture.
        """

        return cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM'])

    def init_training(self, cfg: Dict) -> None:
        """
        Initialize training, including support for the test data loader.

        Args:
            cfg (Dict): Configuration dictionary.
        """

        super().init_training(cfg)
        if hasattr(cfg, 'TEST'):
            self.init_test(cfg)

    @master_only
    def init_test(self, cfg: Dict) -> None:
        """
        Initialize the test data loader and related settings.

        Args:
            cfg (Dict): Configuration dictionary.
        """

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test_time', 'test', '{:.2f} (s)', plt=False)

    @abstractmethod
    def build_train_dataset(self, cfg: Dict) -> Dataset:
        """It must be implement to build dataset for training.

        Args:
            cfg (Dict): config

        Returns:
            train dataset (Dataset)
        """

        pass

    def build_val_dataset(self, cfg: Dict) -> Dataset:
        """It can be implement to build dataset for validation (not necessary).

        Args:
            cfg (Dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    def build_test_dataset(self, cfg: Dict) -> Dataset:
        """
        Build the test dataset.

        Args:
            cfg (Dict): Configuration dictionary.

        Returns:
            Dataset: The test dataset.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('build_test_dataset method must be implemented.')

    def build_test_data_loader(self, cfg: Dict) -> DataLoader:
        """
        Build the test data loader.

        Args:
            cfg (Dict): Configuration dictionary.

        Returns:
            DataLoader: The test data loader.
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    def on_epoch_end(self, epoch: int) -> None:
        """
        Callback at the end of each epoch to handle validation and testing.

        Args:
            epoch (int): The current epoch number.
        """

        # print training meters
        self.print_epoch_meters('train')
        # plot training meters to TensorBoard
        self.plt_epoch_meters('train', epoch)
        # perform validation if configured
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # perform testing if configured
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_pipeline(train_epoch=epoch)
        # save the model checkpoint
        self.save_model(epoch)
        # reset epoch meters
        self.reset_epoch_meters()

    @torch.no_grad()
    @master_only
    def test_pipeline(self, cfg: Optional[Dict] = None, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> None:
        """
        The complete test process.

        Args:
            cfg (Dict, optional): Configuration dictionary. Defaults to None.
            train_epoch (int, optional): Current epoch during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.
        """

        if train_epoch is None and cfg is not None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # execute the test process
        self.test(train_epoch=train_epoch, save_results=save_results, save_metrics=save_metrics)

        test_end_time = time.time()
        self.update_epoch_meter('test_time', test_end_time - test_start_time)

        self.print_epoch_meters('test')
        if train_epoch is not None:
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        # logging here for intuitiveness
        if save_results:
            self.logger.info(f'Test results saved to {os.path.join(self.ckpt_save_dir, "test_results.npz")}.')
        if save_metrics:
            self.logger.info(f'Test metrics saved to {os.path.join(self.ckpt_save_dir, "test_metrics.json")}.')

        self.on_test_end()

    @master_only
    def on_test_start(self) -> None:
        """Callback at the start of testing."""

        pass

    @master_only
    def on_test_end(self) -> None:
        """Callback at the end of testing."""

        pass

    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> None:
        """
        Define the details of the testing process.

        Args:
            train_epoch (int, optional): Current epoch during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('test method must be implemented.')
