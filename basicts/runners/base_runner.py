from abc import abstractmethod
import os
import time
import setproctitle
from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from easytorch import Runner
from easytorch.utils.dist import master_only
from easytorch.utils import get_local_rank, is_master, master_only
from easytorch.core.data_loader import build_data_loader
from basicts.data.transforms import *
from easytorch.utils.timer import TimePredictor

class BaseRunner(Runner):
    def __init__(self, cfg: dict):
        """an expanded easytorch runner for time series models.
            details:
                - support test loader and test process
                - support gradient clip
                - support setup_graph for the models acting like tensorflow
                - move train/validate data loop outside the `train`/`validate` function
        
        Args:
            cfg (dict): all in one configurations
        """
        super().__init__(cfg)

        self.val_interval   = cfg['VAL'].get('INTERVAL', 1)             # validate every `val_interval` epoch
        self.test_interval  = cfg['TEST'].get('INTERVAL', 1)             # test every `test_interval` epoch

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None

        # set proctitle
        proctitle_name = "{0}({1})".format(cfg['MODEL'].get("NAME", " "), cfg.get("DATASET_NAME", " "))
        setproctitle.setproctitle("{0}@BasicTS".format(proctitle_name))

        # Note: other modules, like model, optim, scheduler and so on, have been defined in the super().__init__() function.

    def build_train_data_loader(self, cfg: dict) -> DataLoader:
        train_data_loader = super().build_train_data_loader(cfg)
        if cfg['TRAIN'].get('SETUP_GRAPH', False):
            for data in train_data_loader:
                self.setup_graph(data)
                break
        return train_data_loader

    @staticmethod
    def build_test_dataset(cfg: dict):
        """It can be implemented to a build dataset for validation (not necessary).

        Args:
            cfg (dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    def build_test_data_loader(self, cfg: dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (dict): config

        Returns:
            val data loader (DataLoader)
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    def setup_graph(self, data):
        pass

    def init_training(self, cfg: dict):
        """Initialize training

        Args:
            cfg (dict): config
        """
        super().init_training(cfg)
        # init test
        if hasattr(cfg, 'TEST'):
            self.init_test(cfg)
    
    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """

        # print train meters
        self.print_epoch_meters('train')
        # tensorboard plt meters
        self.plt_epoch_meters('train', epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_main(train_epoch=epoch)
        # save model
        if is_master():
            self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    @torch.no_grad()
    @master_only
    def test_main(self, cfg: dict = None, train_epoch: int = None):
        """test model.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test
        self.test(train_epoch=train_epoch)

        test_end_time = time.time()
        self.update_epoch_meter('test_time', test_start_time - test_end_time)
        # print test meters
        self.print_epoch_meters('test')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        self.on_test_end()

    @master_only
    def init_test(self, cfg: dict):
        """Initialize test

        Args:
            cfg (dict): config
        """

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test_time', 'test', '{:.2f} (s)', plt=False)

    @master_only
    def on_test_start(self):
        """Callback at the start of testing.
        """
        pass

    @master_only
    def on_test_end(self):
        """Callback at the end of testing.
        """
        pass

    def test(self, train_epoch: int = None):
        """It can be implemented to define testing details (not necessary).

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        raise NotImplementedError()

    def train(self, cfg: dict):
        """Train model.

        Train process:
        [init_training]
        for in train_epoch
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]

        Args:
            cfg (dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            data_iter = tqdm(self.train_data_loader) if get_local_rank() == 0 else self.train_data_loader

            # data loop
            self.train_data_loop(data_iter=data_iter, epoch=epoch)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train_time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end()

    def train_data_loop(self, data_iter: tqdm, epoch: int):
        """train data loop

        Args:
            data_iter (tqdm): data iterator
            epoch (int): epoch number
        """
        for iter_index, data in enumerate(data_iter):
            loss = self.train_iters(data, epoch, iter_index)
            if loss is not None:
                self.backward(loss)

    @abstractmethod
    def train_iters(self, data: Union[torch.Tensor, Tuple], epoch: int, iter_index: int) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            data (torch.Tensor or tuple): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """

        pass

    def validate_data_loop(self, train_epoch: int = None):
        """validate data loop

        Args:
            train_epoch (int, optional): current epoch if in training process, else None. Defaults to None.
        """
        for iter_index, data in enumerate(self.val_data_loader):
            self.val_iters(data, train_epoch=train_epoch, iter_index=iter_index)

    def val_iters(self, data: Union[torch.Tensor, Tuple], train_epoch: int, iter_index: int):
        """It can be implemented to define validating details (not necessary).

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process, else None. Defaults to None.
            iter_index (int): current iter.
        """

        raise NotImplementedError()

    @torch.no_grad()
    @master_only
    def validate(self, cfg: dict = None, train_epoch: int = None):
        """Validate model.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init validation if not in training process
        if train_epoch is None:
            self.init_validation(cfg)

        self.on_validating_start(train_epoch=train_epoch)

        val_start_time = time.time()
        self.model.eval()

        # val loop
        self.validate_data_loop(train_epoch=train_epoch)

        val_end_time = time.time()
        self.update_epoch_meter('val_time', val_end_time - val_start_time)
        # print val meters
        self.print_epoch_meters('val')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end(train_epoch=train_epoch)
