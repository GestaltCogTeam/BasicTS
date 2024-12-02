import logging
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import setproctitle
import torch
from easytorch.config import get_ckpt_save_dir
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp
from easytorch.core.meter_pool import MeterPool
from easytorch.device import to_device
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import get_dataset_name
from . import optim


class BaseEpochRunner(metaclass=ABCMeta):
    """
    A base runner that uses epoch as the fundamental training unit.
    This is a general runner.

    Other Features:
        - support torch.compile
        - support early stopping
    """

    # region Initialization Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initialization Functions:                                                             #
    # Essential components setup and initialization for training, validation,               #
    # and testing processes. Includes the following methods:                                #
    #   - __init__: Class initializer                                                       #
    #   - define_mode: Set training mode (train/val/test)                                   #
    #   - build_model: Construct model architecture                                         #
    #   - build_optim: Setup optimizer                                                      #
    #   - build_lr_scheduler: Initialize learning rate scheduler                            #
    #   - build_train_data_loader / build_val_data_loader / build_test_data_loader          #
    #   - build_train_dataset / build_val_dataset / build_test_dataset                      #
    #   - init_training / init_validation / init_test                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self, cfg: Dict) -> None:
        """
        Initialize the BaseRunner.
            - Initialize logger
            - Set environment variables (number of gpus, seeds, cuda related settings, and so on)
            - Define parameters
            - Create save directory
            - Define model
            - Define optimizer and learning rate scheduler
            - Define dataloaders
            - Define meters and tensorboard writer
            - Initialize early stopping

        Args:
            cfg (Dict): Configuration dictionary containing all relevant settings.
        """

        # default logger
        self.logger = get_logger('easytorch')

        # set env
        set_env(cfg.get('ENV', {}))
        # ensure compatibility with higher versions of EasyTorch
        self.to_running_device = to_device

        # param
        self.model_name = cfg['MODEL.NAME']
        self.ckpt_save_dir = get_ckpt_save_dir(cfg)
        self.logger.info('Set ckpt save dir: \'{}\''.format(self.ckpt_save_dir))
        self.ckpt_save_strategy = None
        self.num_epochs = None
        self.start_epoch = None

        self.val_interval = cfg.get('VAL', {}).get('INTERVAL', 1)
        self.test_interval = cfg.get('TEST', {}).get('INTERVAL', 1)

        # create checkpoint save dir
        if not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)

        # create model
        self.model = self.build_model(cfg)

        # declare optimizer and lr_scheduler
        self.optim = None
        self.scheduler = None

        # declare data loaders
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # declare meter pool
        self.meter_pool = None

        # declare tensorboard_writer
        self.tensorboard_writer = None


        # support early stopping
        # NOTE: If the project has been stopped early and its configuration is rerun,
        #           training will resume from the last saved checkpoint.
        #       This feature is designed primarily for the convenience of users,
        #           allowing them to continue training seamlessly after an interruption.
        self.early_stopping_patience = cfg.get('TRAIN', {}).get('EARLY_STOPPING_PATIENCE', None)
        self.current_patience = self.early_stopping_patience
        assert self.early_stopping_patience is None or self.early_stopping_patience > 0, 'Early stopping patience must be a positive integer.'

        # set process title
        proctitle_name = f"{cfg['MODEL'].get('NAME')}({get_dataset_name(cfg)})"
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

    def build_model(self, cfg: Dict) -> nn.Module:
        """Build model.

        Initialize model by calling ```self.define_model```,
        Moves model to the GPU.

        If DDP is initialized, initialize the DDP wrapper.

        Args:
            cfg (Dict): config

        Returns:
            model (nn.Module)
        """

        self.logger.info('Building model.')
        model = self.define_model(cfg)
        model = self.to_running_device(model)

        # complie model
        if cfg.get('TRAIN.COMPILE_MODEL', False):
            # get current torch version
            current_version = torch.__version__
            # torch.compile() is only available in torch>=2.0
            if version.parse(current_version) >= version.parse('2.0'):
                self.logger.info('Compile model with torch.compile')
                model = torch.compile(model)
            else:
                self.logger.warning(f'torch.compile requires PyTorch 2.0 or higher. Current version: {current_version}. Skipping compilation.')

        # DDP
        if torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=cfg.get('MODEL.DDP_FIND_UNUSED_PARAMETERS', False)
            )
        return model

    def build_optim(self, optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:
        """Build optimizer from `optim_cfg`.

        Args:
            optim_cfg (Dict): optimizer config
            model (nn.Module): model

        Returns:
            optim.Optimizer: optimizer
        """

        # TODO: support other optimizers here
        return optim.build_optim(optim_cfg, model)

    def build_lr_scheduler(self, cfg: Dict) -> None:
        """
        Initialize lr_scheduler

        Args:
            cfg (Dict): config
        """
        # create lr_scheduler
        if cfg.has('TRAIN.LR_SCHEDULER'):
            self.scheduler = optim.build_lr_scheduler(cfg['TRAIN.LR_SCHEDULER'], self.optim)
            self.logger.info('Set lr_scheduler: {}'.format(self.scheduler))
            self.register_epoch_meter('train/lr', 'train', '{:.2e}')

    def build_train_data_loader(self, cfg: Dict) -> DataLoader:
        """Build train dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader``` or
        ```build_data_loader_ddp``` when DDP is initialized

        Args:
            cfg (Dict): config

        Returns:
            train data loader (DataLoader)
        """

        self.logger.info('Building training data loader.')
        dataset = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return build_data_loader(dataset, cfg['TRAIN.DATA'])

    @abstractmethod
    def build_train_dataset(self, cfg: Dict) -> Dataset:
        """It must be implement to build dataset for training.

        Args:
            cfg (Dict): config

        Returns:
            train dataset (Dataset)
        """

        pass

    def build_val_data_loader(self, cfg: Dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (Dict): config

        Returns:
            val data loader (DataLoader)
        """

        self.logger.info('Building val data loader.')
        dataset = self.build_val_dataset(cfg)
        return build_data_loader(dataset, cfg['VAL.DATA'])

    @staticmethod
    def build_val_dataset(cfg: Dict) -> Dataset:
        """It can be implement to build dataset for validation (not necessary).

        Args:
            cfg (Dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

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

    def init_training(self, cfg: Dict) -> None:
        """
        Initialize training, including support for the test data loader.

        Args:
            cfg (Dict): Configuration dictionary.
        """

        self.logger.info('Initializing training.')

        # init training param
        self.num_epochs = cfg['TRAIN.NUM_EPOCHS']
        self.start_epoch = 0
        self.ckpt_save_strategy = cfg.get('TRAIN.CKPT_SAVE_STRATEGY')
        self.best_metrics = {}
        self.clip_grad_param = cfg.get('TRAIN.CLIP_GRAD_PARAM')
        if self.clip_grad_param is not None:
            self.logger.info('Set clip grad, param: {}'.format(self.clip_grad_param))

        # train data loader
        self.train_data_loader = self.build_train_data_loader(cfg)
        self.register_epoch_meter('train/time', 'train', '{:.2f} (s)', plt=False)

        # create optim
        self.optim = self.build_optim(cfg['TRAIN.OPTIM'], self.model)
        self.logger.info('Set optim: {}'.format(self.optim))

        # create lr_scheduler
        self.build_lr_scheduler(cfg)

        # fine tune
        if cfg.has('TRAIN.FINETUNE_FROM'):
            self.load_model(cfg['TRAIN.FINETUNE_FROM'], cfg.get('TRAIN.FINETUNE_STRICT_LOAD', True))
            self.logger.info('Start fine tuning')

        # resume
        self.load_model_resume()

        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

        # init validation
        if cfg.has('VAL'):
            self.init_validation(cfg)

        if hasattr(cfg, 'TEST'):
            self.init_test(cfg)

    @master_only
    def init_validation(self, cfg: Dict):
        """Initialize validation

        Args:
            cfg (Dict): config
        """

        self.logger.info('Initializing validation.')
        self.val_interval = cfg.get('VAL.INTERVAL', 1)
        self.val_data_loader = self.build_val_data_loader(cfg)
        self.register_epoch_meter('val/time', 'val', '{:.2f} (s)', plt=False)

    @master_only
    def init_test(self, cfg: Dict) -> None:
        """
        Initialize the test data loader and related settings.

        Args:
            cfg (Dict): Configuration dictionary.
        """

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test/time', 'test', '{:.2f} (s)', plt=False)

    # endregion Initialization Functions

    # region Entries
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Entry Points:                                                                         #
    # Control the overall process for training, validation, and testing.                    #
    #   - train: Entry point for the training process.                                      #
    #   - validate: Entry point for the validation process.                                 #
    #   - test_pipeline: Entry point for the evaluation process.                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def train(self, cfg: Dict) -> None:
        """Train model. 
            The start point of training process.

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
            cfg (Dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        epoch_index = 0
        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1

            # early stopping check
            if self.check_early_stopping():
                break

            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            data_loader = tqdm(self.train_data_loader) if get_local_rank() == 0 \
                                                        else self.train_data_loader

            # data loop
            for iter_index, data in enumerate(data_loader):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train/time', epoch_end_time - epoch_start_time)
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

        self.on_training_end(cfg=cfg, train_epoch=epoch_index + 1)

    @torch.no_grad()
    @master_only
    def validate(self, cfg: Dict = None, train_epoch: Optional[int] = None):
        """Validate model.

        Args:
            cfg (Dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init validation if not in training process
        if train_epoch is None:
            self.init_validation(cfg)

        self.logger.info('Start validation.')

        self.on_validating_start(train_epoch)

        val_start_time = time.time()
        self.model.eval()

        # tqdm process bar
        data_iter = tqdm(self.val_data_loader)

        # val loop
        for iter_index, data in enumerate(data_iter):
            self.val_iters(iter_index, data)

        val_end_time = time.time()
        self.update_epoch_meter('val/time', val_end_time - val_start_time)
        # print val meters
        self.print_epoch_meters('val')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end(train_epoch)

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
        self.update_epoch_meter('test/time', test_end_time - test_start_time)

        self.print_epoch_meters('test')
        if train_epoch is not None:
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        # logging here for intuitiveness
        if save_results:
            self.logger.info(f'Test results saved to {os.path.join(self.ckpt_save_dir, "test_results.npz")}.')
        if save_metrics:
            self.logger.info(f'Test metrics saved to {os.path.join(self.ckpt_save_dir, "test_metrics.json")}.')

        self.on_test_end()

    # endregion Entries

    # region Main Loops
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Main Loops:                                                                           #
    # Each iteration within training/validation, and the complete test process.             #
    #   - train_iter: A single iteration of the training loop.                              #
    #   - val_iter: A single iteration of the validation loop.                              #
    #   - test: Full evaluation process (distinct from train/val loops).                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @abstractmethod
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """

        pass

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        """It can be implement to define validating detail (not necessary).

        Args:
            iter_index (int): current iter.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
        """

        raise NotImplementedError()

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

    # endregion Main Loops

    # region Hook Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Hook Functions:                                                                       #
    # Called within the entry functions (train, validate, test_pipeline).                   #
    #   - on_epoch_start / on_epoch_end: Hooks for the start/end of each epoch.             #
    #   - on_validating_start / on_validating_end: Hooks for validation start/end.          #
    #   - on_testing_start / on_testing_end: Hooks for test start/end.                      #
    #   - on_training_end: Hook for the end of training process.                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def on_epoch_start(self, epoch: int):
        """Callback at the start of an epoch.

        Args:
            epoch (int): current epoch
        """

        # print epoch num
        self.logger.info('Epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        # update lr meter
        if self.scheduler is not None:
            self.update_epoch_meter('train/lr', self.scheduler.get_last_lr()[0])

        # set epoch for sampler in distributed mode
        # see https://pytorch.org/docs/stable/data.html
        sampler = self.train_data_loader.sampler
        if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
            sampler.set_epoch(epoch)

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

    @master_only
    def on_validating_start(self, train_epoch: Optional[int]):
        """Callback at the start of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        pass

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        pass

    @master_only
    def on_test_start(self) -> None:
        """Callback at the start of testing."""

        pass

    @master_only
    def on_test_end(self) -> None:
        """Callback at the end of testing."""

        pass

    def on_training_end(self, cfg: Dict, train_epoch: Optional[int] = None):
        """Callback at the end of the training process.
        
        Args:
            cfg (Dict): Configuration.
            train_epoch (Optional[int]): End epoch if in training process.
        """

        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

        if hasattr(cfg, 'TEST'):
            # evaluate the best model on the test set
            best_model_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_val_{}.pt'.format(self.model_name, self.target_metrics.replace('/', '_'))
            )
            self.logger.info('Evaluating the best model on the test set.')
            self.load_model(ckpt_path=best_model_path, strict=True)
            self.test_pipeline(cfg=cfg, train_epoch=train_epoch, save_metrics=True, save_results=True)

    # endregion Hook Functions

    # region Misc Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous Functions:                                                              #
    # Utility functions for model loading, saving, and other tasks.                         #
    #   - load_model / load_model_resume: Load or resume model.                             #
    #   - backward: Backward loss.                                                          #
    #   - save_model / save_best_model: Save models and best checkpoints.                   #
    #   - init_logger: Initialize logger for process tracking.                              #
    #   - get_ckpt_path: Retrieve checkpoint path.                                          #
    #   - check_early_stopping: Check for early stopping conditions.                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load model state dict.
        if param `ckpt_path` is None, load the last checkpoint in `self.ckpt_save_dir`,
        else load checkpoint from `ckpt_path`

        Args:
            ckpt_path (str, optional): checkpoint path, default is None
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e

    def load_model_resume(self, strict: bool = True):
        """Load last checkpoint in checkpoint save dir to resume training.

        Load model state dict.
        Load optimizer state dict.
        Load start epoch and set it to lr_scheduler.

        Args:
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if checkpoint_dict.get('best_metrics') is not None:
                self.best_metrics = checkpoint_dict['best_metrics']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('Resume training')
        except (IndexError, OSError, KeyError):
            pass

    def backward(self, loss: torch.Tensor):
        """Backward and update params.

        Args:
            loss (torch.Tensor): loss
        """

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad_param is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
        self.optim.step()

    @master_only
    def save_model(self, epoch: int):
        """Save checkpoint every epoch.

        checkpoint format is {
            'epoch': current epoch ([1, num_epochs]),
            'model_state_dict': state_dict of model,
            'optim_state_dict': state_dict of optimizer
        }

        Decide whether to delete the last checkpoint by the checkpoint save strategy.

        Args:
            epoch (int): current epoch.
        """

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics
        }

        # backup last epoch
        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # save ckpt
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10 epoch or in the end
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        """Save the best model while training.

        Examples:
            >>> def on_validating_end(self, train_epoch: Optional[int]):
            >>>     if train_epoch is not None:
            >>>         self.save_best_model(train_epoch, 'val/loss', greater_best=False)

        Args:
            epoch (int): current epoch.
            metric_name (str): metric name used to measure the model, must be registered in `epoch_meter`.
            greater_best (bool, optional): `True` means greater value is best, such as `acc`
                `False` means lower value is best, such as `loss`. Defaults to True.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience # reset patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def init_logger(self, logger: logging.Logger = None, logger_name: str = None,
                    log_file_name: str = None, log_level: int = logging.INFO) -> None:
        """Initialize logger.

        Args:
            logger (logging.Logger, optional): specified logger.
            logger_name (str, optional): specified name of logger.
            log_file_name (str, optional): logger file name.
            log_level (int, optional): log level, default is INFO.
        """

        if logger is not None:
            self.logger = logger
        elif logger_name is not None:
            if log_file_name is not None:
                log_file_name = '{}_{}.log'.format(log_file_name, time.strftime('%Y%m%d%H%M%S', time.localtime()))
                log_file_path = os.path.join(self.ckpt_save_dir, log_file_name)
            else:
                log_file_path = None
            self.logger = get_logger(logger_name, log_file_path, log_level)
        else:
            raise TypeError('At least one of logger and logger_name is not None')

    def get_ckpt_path(self, epoch: int) -> str:
        """Get checkpoint path.

        The format is "{ckpt_save_dir}/{model_name}_{epoch}"

        Args:
            epoch (int): current epoch.

        Returns:
            checkpoint path (str)
        """

        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    def check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.early_stopping_patience is not None and self.current_patience <= 0:
            self.logger.info('Early stopping.')
            return True
        return False

    # endregion Misc Functions

    # region meters and tensorboard
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Meters and Tensorboard Functions:                                                     #
    # Tools for tracking, updating, and visualizing training metrics.                       #
    #   - register_epoch_meter: Register metrics.                                       #
    #   - update_epoch_meter: Update metrics.                                           #
    #   - print_epoch_meters: Print current metrics.                                    #
    #   - plt_epoch_meters: Plot metrics.                                               #
    #   - reset_epoch_meters: Reset metrics.                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True) -> None:
        if self.meter_pool is None:
            self.meter_pool = MeterPool()
        self.meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_epoch_meter(self, name, value, n=1) -> None:
        self.meter_pool.update(name, value, n)

    @master_only
    def print_epoch_meters(self, meter_type) -> None:
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_epoch_meters(self, meter_type, step) -> None:
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer)

    @master_only
    def reset_epoch_meters(self) -> None:
        self.meter_pool.reset()

    # endregion meters and tensorboard
