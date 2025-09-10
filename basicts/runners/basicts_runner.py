import inspect
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import setproctitle
import torch
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)
from easytorch.device import to_device
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             get_world_size, is_master, master_only)
from easytorch.utils.data_prefetcher import DataLoaderX
from easytorch.utils.env import get_rank, set_tf32_mode, setup_determinacy
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basicts.metrics import ALL_METRICS
from basicts.scaler import BasicTSScaler

from ..data import TimeSeriesInferenceDataset
from ..utils import BasicTSMode, MeterPool, RunnerStatus
from .callback import BasicTSCallbackHandler
# from .distributed import distributed
from .taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.configs import BasicTSConfig


class BasicTSRunner:
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

    def __init__(self, cfg: 'BasicTSConfig') -> None:
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

        # config
        self.cfg: 'BasicTSConfig' = cfg
        # default logger
        self.logger = get_logger('BasicTS')
        # status
        self.status: RunnerStatus = RunnerStatus.INITIALIZING
        # task flow
        self.taskflow: BasicTSTaskFlow = cfg.taskflow
        # callback handler
        self.callback_handler = BasicTSCallbackHandler(cfg.callbacks)

        # initialization flag
        self.is_train_initialized = False
        self.is_val_initialized = False
        self.is_test_initialized = False

        # training unit
        self.training_unit: Literal['epoch', 'step'] = None

        # set environment variables
        self.set_env(cfg)
        # ensure compatibility with higher versions of EasyTorch
        self.to_running_device: Callable = to_device
        # create meter pool
        self.meter_pool: MeterPool = MeterPool()
        # create model
        self.model: torch.nn.Module = self.build_model(cfg)
        self.model_name: str = self.model.__class__.__name__

        self.num_epochs = None
        self.num_steps = None
        self.start_epoch = None
        self.evaluation_horizons = None
        self.save_results = cfg.save_results

        # define loss function
        self.loss = cfg.loss

        # declare optimizer and lr_scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # automatic mixed precision (amp)
        model_dtype: str | torch.dtype = cfg.get('model_dtype', 'float32')
        if isinstance(self.model_dtype, str):
            self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[model_dtype]
        else:
            self.ptdtype = model_dtype
        self.use_amp = self.ptdtype in [torch.bfloat16, torch.float16]
        if self.use_amp: assert cfg.gpus is not None, 'AMP only supports CUDA.'
        self.amp_ctx = torch.amp.autocast(device_type='cuda', dtype=self.ptdtype, enabled=self.use_amp)
        # GradScaler will scale up gradients and some of them might become inf, which may cause lr_scheduler throw incorrect warning information. See:
        # https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
        self.amp_scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # declare data loaders
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # declare scaler
        self.scaler: Optional[BasicTSScaler] = self.cfg.scaler

        # define metrics
        self.metrics = {k: ALL_METRICS[k] for k in cfg.metrics if k in ALL_METRICS}
        self.target_metric = cfg.target_metric
        self.metrics_best = cfg.best_metric
        assert self.target_metric in self.metrics or self.target_metric == 'loss', f'Target metric {self.target_metric} not found in metrics.'
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'
        # handle null values in datasets, e.g., 0.0 or np.nan.
        self.null_val = cfg.dataset.null_val

        # declare tensorboard_writer
        self.tensorboard_writer = None

        self.ckpt_save_dir: str = self._get_ckpt_save_dir(cfg)
        self.logger.info('Set ckpt save dir: \'{}\''.format(self.ckpt_save_dir))
        self.ckpt_save_strategy = None
        # create checkpoint save dir
        if not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)

        # TODO: define a Control class
        self.should_training_stop = False
        self.should_optimizer_step = True

        # set process title
        proctitle_name = f"{cfg.model.__class__.__name__}({cfg.dataset.name})"
        setproctitle.setproctitle(f'{proctitle_name}@BasicTS')

    @master_only
    def _init_train(self):
        self.logger.info('Initializing training.')

        # init training param
        self.num_epochs = self.cfg.num_epochs
        self.start_epoch = 0
        self.ckpt_save_strategy = self.cfg.ckpt_save_strategy
        if self.cfg.num_epochs is not None and not hasattr(self.cfg, 'num_steps'):
            self.training_unit = 'epoch'
        elif self.cfg.num_steps is not None and not hasattr(self.cfg, 'num_epochs'):
            self.training_unit = 'step'
        else:
            raise ValueError('`num_epochs` and `num_steps` cannot be set at the same time.')
        self.best_metrics = {}
        self.clip_grad_param = self.cfg.clip_grad_param
        self.train_data_loader = self._build_data_loader(self.cfg, BasicTSMode.TRAIN)
        self.register_meter('train/time', 'train', '{:.2f} (s)', plt=False)
        if self.scaler is not None:
            self.scaler.fit(self.train_data_loader.dataset.data)
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        if self.use_amp:
            self.register_meter('train/amp_scale', 'train', '{:.4f}')
        # fine tune
        if self.cfg.finetune_from is not None:
            self.load_model(self.cfg.finetune_from, self.cfg.strict_load)
            self.logger.info('Start fine tuning')
        # resume
        self.load_model_resume()
        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )
        # log
        if self.clip_grad_param is not None:
            self.logger.info('Set clip grad, param: {}'.format(self.clip_grad_param))
        self.logger.info('Set optim: {}'.format(self.optimizer))
        if self.cfg.lr_scheduler is not None:
            self.logger.info('Set lr_scheduler: {}'.format(self.lr_scheduler))
            self.register_meter('train/lr', 'train', '{:.2e}')

    @master_only
    def _init_validation(self):
        """Initialize validation"""

        self.val_interval = self.cfg.val_interval
        self.val_data_loader = self._build_data_loader(self.cfg, BasicTSMode.VAL)
        self.register_meter('val/time', 'val', '{:.2f} (s)', plt=False)
        self.register_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_meter(f'val/{key}', 'val', '{:.4f}')

    @master_only
    def _init_test(self) -> None:
        """
        Initialize the test data loader and related settings.

        Args:
            cfg (BasicTSConfig): Configuration dictionary.
        """

        self.test_interval = self.cfg.test_interval
        self.test_data_loader = self._build_data_loader(self.cfg, BasicTSMode.TEST)
        self.register_meter('test/time', 'test', '{:.2f} (s)', plt=False)
        self.register_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_meter(f'test/{key}', 'test', '{:.4f}')

    @master_only
    def _init_inference(self, cfg: Dict, input_data: Union[str, list]) -> None:
        """
        Initialize the inference data loader and related settings.

        Args:
            cfg (Dict): Configuration dictionary.
            input_data (Union[str, list]): The input data file path or data list for inference.
        """

        self.inference_dataset = self.build_dataset(TimeSeriesInferenceDataset, cfg, input_data)
        self.inference_dataset_loader = DataLoader(self.inference_dataset, batch_size=1, shuffle=False)
        self.register_meter('inference/time', 'inference', '{:.2f} (s)', plt=False)

    # endregion Initialization Functions

    # region Entries
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Entry Points:                                                                         #
    # Control the overall process for training, validation, and evaluation.                 #
    #   - train: Entry point for the training process.                                      #
    #   - validate: Entry point for the validation process.                                 #
    #   - eval: Entry point for the evaluation process.                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # TODO: move distributed training to runner
    # @distributed(default_node_num=1, default_device_num=self.gpu_num)
    def train(self):
        self.on_train_start()
        self.callback_handler.trigger('on_train_start', self)
        self._train_loop()
        self.callback_handler.trigger('on_train_end', self)
        self.on_train_end()
        # evaluate the best model on the test set
        best_model_path = os.path.join(
            self.ckpt_save_dir,
            '{}_best_val_{}.pt'.format(self.model_name, self.target_metric.replace('/', '_'))
        )
        self.logger.info('Evaluating the best model on the test set.')
        self.eval(best_model_path)

    @torch.no_grad()
    def validate(self, train_step: Optional[int] = None, train_epoch: Optional[int] = None):
        """Validate model.

        Args:
            cfg (Dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        self.on_validate_start()
        self.callback_handler.trigger('on_validate_start', self)
        self.logger.info('Start validation.')
        self.status = RunnerStatus.VALIDATING
        self.model.eval()
        val_start_time = time.time()
        self._eval_loop(BasicTSMode.VAL)
        val_end_time = time.time()
        self.update_meter('val/time', val_end_time - val_start_time)
        self.callback_handler.trigger('on_validate_end', self, train_step=train_step, train_epoch=train_epoch)
        self.on_validate_end(train_step, train_epoch)

    @torch.no_grad()
    def eval(self, ckpt_path: Optional[str] = None) -> None:
        """
        The complete evaluation process.

        Args:
            train_epoch (int, optional): Current epoch during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.
        """
        self.on_eval_start(ckpt_path)
        self._test(None, None, BasicTSMode.EVAL)
        self.on_eval_end()

    @torch.no_grad()
    def _test(self, train_step: Optional[int] = None, train_epoch: Optional[int] = None, \
              mode: BasicTSMode = Optional[BasicTSMode.TEST]) -> None:
        """
        The complete test process. It should not be directly called.

        Args:
            train_epoch (int, optional): Current epoch during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.
        """

        self.on_test_start()
        self.callback_handler.trigger('on_test_start', self)
        # self.logger.info('Start testing.')
        self.model.eval()
        self.status = RunnerStatus.TESTING if mode == BasicTSMode.TEST else RunnerStatus.EVALUATING
        test_start_time = time.time()
        self._eval_loop(mode)
        test_end_time = time.time()
        self.update_meter('test/time', test_end_time - test_start_time)
        self.callback_handler.trigger('on_test_end', self)
        self.on_test_end(train_step, train_epoch)

    # endregion Entries

    # region Main Loops
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Main Loops:                                                                           #
    # The loops for train, validation, and test process.                                    #
    #   - _train_loop: Loop for training.                                                   #
    #   - _eval_loop: Loop for validation and test.                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _train_loop(self) -> None:
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

        if self.training_unit == 'epoch':
            self.num_steps = (self.num_epochs - self.start_epoch) * self.step_per_epoch
            self.start_step = 0
            step_pbar = None

        else: # training_unit == 'step'
            step_pbar = tqdm(range(self.start_step, self.num_steps),
                             initial=self.start_step + 1,
                             total=self.num_steps,
                             mininterval=0)

        # train time predictor
        self.train_time_predictor = TimePredictor(self.start_step, self.num_steps)

        # training loop
        step = self.start_step
        while step < self.num_steps:

            if self.should_training_stop:
                break

            data_loader = self.train_data_loader

            # update epoch
            self.epoch = step // self.step_per_epoch + self.start_epoch + 1
            # a new epoch
            if self.training_unit == 'epoch' and step % self.step_per_epoch == 0:
                self.on_epoch_start(self.epoch)
                self.callback_handler.trigger('on_epoch_start', self, epoch=self.epoch)
                # epoch pbar
                data_loader = tqdm(data_loader) if get_local_rank() == 0 else data_loader

            # data loop
            for data in data_loader:

                if self.should_training_stop:
                    break

                self.status = RunnerStatus.TRAINING
                self.on_step_start(step)
                self.callback_handler.trigger('on_step_start', self, epoch=self.epoch, step=step)
                self.model.train()
                step_start_time = time.time()
                data = self.taskflow.preprocess(self, data) # task specific preprocess
                with self.amp_ctx:
                    forward_return = self._forward(data, step, self.epoch)
                    self.callback_handler.trigger('on_compute_loss', self, epoch=self.epoch, step=step, data=data, forward_return=forward_return)
                    loss = self._metric_forward(self.loss, forward_return)
                loss_weight = self.taskflow.get_weight(forward_return) # task specific metric weight for averaging
                self.update_meter('train/loss', loss.item(), loss_weight)
                self.callback_handler.trigger('on_backward', self, loss=loss)
                with (self.model.no_sync() if hasattr(self.model, 'no_sync') else nullcontext()):
                    self.amp_scaler.scale(loss).backward() # if not use_amp, it equals to loss.backward()
                if self.should_optimizer_step:
                    self.callback_handler.trigger('on_optimizer_step', self)
                    self._optimizer_step()
                forward_return = self.taskflow.postprocess(self, forward_return) # task specific postprocess
                # update metrics meter
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = self._metric_forward(metric_fn, forward_return)
                    metric_weight = self.taskflow.get_weight(forward_return) # task specific metric weight for averaging
                    self.update_meter(f'train/{metric_name}', metric_value.item(), metric_weight)

                self.callback_handler.trigger('on_step_end', self, step=step)
                self.on_step_end(step)
                step_end_time = time.time()
                self.update_meter('train/time', step_end_time - step_start_time)

                # step end
                step += 1
                # update progress bar
                if step_pbar is not None:
                    step_pbar.update()
                # check if training should stop
                if step >= self.num_steps:
                    self.should_training_stop = True

            # when training unit is epoch, call on epoch end
            if self.training_unit == 'epoch':
                self.callback_handler.trigger('on_epoch_end', self, epoch=self.epoch, step=step)
                self.on_epoch_end(self.epoch, step)

    def _eval_loop(self, mode: BasicTSMode):
        """evaluate model.
        """

        # tqdm process bar
        leave = False if self.training_unit == 'step' and mode != BasicTSMode.EVAL else True
        if mode == BasicTSMode.VAL:
            data_iter = tqdm(self.val_data_loader, leave=leave)
            meter_type = 'val'
        else:
            data_iter = tqdm(self.test_data_loader, leave=leave)
            meter_type = 'test'

        # eval loop
        for step, data in enumerate(data_iter):
            data = self.taskflow.preprocess(self, data) # task specific preprocess
            # TODO: consider using amp for validation
            # with self.ctx:
            forward_return = self._forward(data, step=step, train=False)
            self.callback_handler.trigger('on_compute_loss', self, forward_return=forward_return)
            # compute validation loss
            loss = self._metric_forward(self.cfg.loss, forward_return)
            loss_weight = self.taskflow.get_weight(forward_return) # task specific metric weight for averaging
            self.update_meter(f'{meter_type}/loss', loss.item(), loss_weight)
            forward_return = self.taskflow.postprocess(self, forward_return)
            if mode == BasicTSMode.EVAL:
                self._save_results(step, forward_return)
            # update metrics meter
            for metric_name, metric_item in self.metrics.items():
                metric_value = self._metric_forward(metric_item, forward_return)
                metric_weight = self.taskflow.get_weight(forward_return) # task specific metric weight for averaging
                self.update_meter(f'{meter_type}/{metric_name}', metric_value.item(), metric_weight)

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

    def on_train_start(self) -> None:
        """
        Initialize training, including support for the test data loader.
        """

        self._init_train()
        self.is_train_initialized = True
        # init validation
        if hasattr(self.cfg, 'val_interval'):
            self._init_validation()
            self.is_val_initialized = True

        if hasattr(self.cfg, 'test_interval'):
            self._init_test()
            self.is_test_initialized = True

        # TODO: move setup graph to callback
        # if self.need_setup_graph:
        #     self.setup_graph(cfg=cfg, train=True)
        #     self.need_setup_graph = False

        self._count_parameters()

        self.register_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_meter(f'train/{key}', 'train', '{:.4f}')

    def on_train_end(self) -> None:
        """Callback at the end of the training process.
        
        Args:
            cfg (Dict): Configuration.
            train_epoch (Optional[int]): End epoch if in training process.
        """

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

    def on_validate_start(self):
        """Callback at the start of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if not self.is_val_initialized:
            self._init_validation()

    def on_validate_end(self, train_step: int, train_epoch: Optional[int] = None):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        # print val meters
        self.print_meters('val')

        greater_best = not self.metrics_best == 'min'
        if self.training_unit == 'epoch' and train_epoch is not None:
            # tensorboard plt meters
            self.plt_meters('val', train_epoch // self.val_interval)
            self._save_best_model(train_epoch, 'val/' + self.target_metric, greater_best=greater_best)
        else: # training_unit is 'step'
            self.plt_meters('val', train_step // self.val_interval)
            self._save_best_model(train_step, 'val/' + self.target_metric, greater_best=greater_best)

    @master_only
    def on_test_start(self) -> None:
        """Callback at the start of testing."""

        if not self.is_test_initialized:
            self._init_test()

    @master_only
    def on_test_end(self, train_step: Optional[int] = None, train_epoch: Optional[int] = None) -> None:
        """Callback at the end of testing."""

        # print test metrics
        self.print_meters('test')
        if self.evaluation_horizons is not None and len(self.evaluation_horizons) > 0:
            self.logger.info(f'Evaluation on horizons: {[h + 1 for h in self.evaluation_horizons]}.')
            for i in self.evaluation_horizons:
                self.print_meters(f'test @ horizon {i+1}')

        if self.training_unit == 'epoch' and train_epoch is not None:
            # tensorboard plt meters
            self.plt_meters('test', train_epoch // self.test_interval)
        if self.training_unit == 'step' and train_step is not None:
            self.plt_meters('test', train_step // self.test_interval)

    def on_eval_start(self, ckpt_path: str) -> None:
        """Callback at the start of evaluation."""

        if not self.is_train_initialized and self.scaler is not None:
            train_dataset = self.cfg.dataset(BasicTSMode.TRAIN)
            self.scaler.fit(train_dataset.data)
        ckpt_path = os.path.join(self.ckpt_save_dir, '{}_best_val_{}.pt'.format(self.model_name, self.target_metric.replace('/', '_')))
        self.load_model(ckpt_path=ckpt_path, strict=True)

    @master_only
    def on_eval_end(self) -> None:
        """Callback at the end of evaluation."""

        # save results to self.ckpt_save_dir/test_results (in eval loop)
        if self.cfg.save_results:
            self.logger.info(f'Test results saved to {os.path.join(self.ckpt_save_dir, "test_results")}.')

        # save metrics results to self.ckpt_save_dir/test_metrics.json
        self._save_metrics()

    def on_epoch_start(self, epoch: int) -> None:
        """Callback at the start of an epoch.

        Args:
            epoch (int): current epoch
        """

        # print epoch num
        self.logger.info('Epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        # set epoch for sampler in distributed mode
        # see https://pytorch.org/docs/stable/data.html
        sampler = self.train_data_loader.sampler
        if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
            sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch: int, step: int) -> None:
        """
        Callback at the end of each epoch to handle validation and testing.

        Args:
            epoch (int): The current epoch number.
        """

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.update_meter('train/lr', self.lr_scheduler.get_last_lr()[0])

        # print training meters
        self.print_meters('train')
        # plot training meters to TensorBoard
        self.plt_meters('train', epoch)
        # perform validation if configured
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # perform testing if configured
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self._test(train_epoch=epoch)
        # save the model checkpoint
        self._save_model(epoch)
        # reset epoch meters
        self.reset_meters()

        # estimate training finish time
        if not self.should_training_stop and self.epoch < self.num_epochs:
            expected_end_time = self.train_time_predictor.get_expected_end_time(step)
            self.logger.info('The estimated training finish time is {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

    def on_step_start(self, step: int) -> None:
        """Called before each step.

        Args:
            step (int): current step
        """

        if self.training_unit == 'step' and step % self.val_interval == 0:
            # print iteration num
            self.logger.info('Iteration {:d} / {:d}'.format(step, self.num_steps))

    def on_step_end(self, step: int) -> None:
        """Called after each step.

        Args:
            epoch (int): current epoch
            step (int): current step
            data (Dict): data
            forward_return (Dict): forward return
        """

        if self.training_unit == 'step':
            # update lr_scheduler per step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.update_meter('train/lr', self.lr_scheduler.get_last_lr()[0])

            # tensorboard plt meters
            # self.plt_iteration_meters('train', iteration, value_type='last')
            if step % self.val_interval == 0:
                # print train meters
                self.print_meters('train')
                # validate
                if self.val_data_loader is not None:
                    self.validate(train_step=step)
                    # save model
                    self._save_model(step)
                    # reset meters
                    self.reset_meters()

            if self.test_interval is not None and step % self.test_interval == 0:
                self._test(step)

    @master_only
    def on_inference_start(self) -> None:
        """Callback at the start of inference."""

        pass

    @master_only
    def on_inference_end(self) -> None:
        """Callback at the end of inference."""

        pass

    # endregion Hook Functions

    # key methods

    def _forward(self, data: Dict, step: int, epoch: Optional[int] = None, train: bool = True) -> Optional[torch.Tensor]:
        """Train model.

        Args:
            epoch (int): current epoch
            iter_index (int): current iter index
            data (Dict): data

        Returns:
            Optional[torch.Tensor]: loss
        """

        # move data to running device
        for k in data.keys():
            data[k] = self.to_running_device(data[k]) if isinstance(data[k], torch.Tensor) else data[k]

        # data must contain 'inputs' and 'target'
        assert 'inputs' in data and 'target' in data, 'data must contain key \'inputs\' and \'target\'.'
        inputs, target = data['inputs'], data['target']

        # For non-training phases, the model should not be able to access target
        target_4_model_input = torch.empty_like(target) if not train else target

        sig = inspect.signature(self.model.forward)
        all_params = list(sig.parameters.keys())
        remaining_params = all_params[2:] # except for inputs and target
        kwargs = {
            k: data[k] if isinstance(data[k], torch.Tensor) else data[k]
            for k in remaining_params if k in data
        }
        if 'batch_seen' in sig.parameters.keys():
            kwargs['batch_seen'] = step
        if 'epoch' in sig.parameters.keys():
            kwargs['epoch'] = epoch
        if 'train' in sig.parameters.keys():
            kwargs['train'] = train

        # Forward pass through the model
        forward_return = self.model(inputs, target_4_model_input, **kwargs)

        # Parse forward return
        if isinstance(forward_return, torch.Tensor):
            forward_return = {'prediction': forward_return}
        # add other keys and values in `data` to `forward_return`
        for k, v in data.items():
            forward_return[k] = v
        return forward_return

    def _metric_forward(self, metric_fn: Callable, forward_return: Dict[str, Any]) -> torch.Tensor:
        covariate_names = inspect.signature(metric_fn).parameters.keys()
        args = {k: v for k, v in forward_return.items() if k in covariate_names}
        if callable(metric_fn):
            metric_value = metric_fn(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_fn)}')
        return metric_value

    def _optimizer_step(self):
        if self.use_amp:
            self.update_meter('train/amp_scale', self.amp_scaler.get_scale())
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def set_env(self, cfg: 'BasicTSConfig'):
        """Setup runtime env, include tf32, seed and determinacy.

        env config template:
        ```
        cfg.tf32 = False
        cfg.seed = 42
        cfg.deterministic = True
        cfg.cudnn_enabled = False
        cfg.cudnn_benchmark = False
        cfg.cudnn_determinstic = True
        ```

        Args:
            cfg (BasicTSConfig): env config.
        """

        # tf32
        set_tf32_mode(cfg.tf32)

        # determinacy
        seed = cfg.seed
        if seed is not None:
            # each rank has different seed in distributed mode
            setup_determinacy(
                seed + get_rank(),
                cfg.deterministic,
                cfg.cudnn_enabled,
                cfg.cudnn_benchmark,
                cfg.cudnn_determinstic
            )

    def build_model(self, cfg: 'BasicTSConfig'):
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
        model = self.to_running_device(cfg.model)

        # complie model
        if cfg.compile_model:
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
                find_unused_parameters=cfg.ddp_find_unused_parameters
            )
        return model

    def _build_data_loader(self, cfg: 'BasicTSConfig', mode: BasicTSMode):
        """Build dataloader from BasicTSConfig

        structure of `data_cfg` is
        {
            'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
            'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
            'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
            'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
            'PREFETCH': (bool, optional) set to ``True`` to use `DataLoaderX` (default: ``False``),
        }

        Args:
            data_cfg (BasicTSConfig): config
            mode (Literal['train', 'val', 'test']): mode

        Returns:
            data loader
        """

        self.logger.info(f'Building {mode} data loader.')
        dataset = cfg.dataset(mode=mode)
        if mode == BasicTSMode.TRAIN:
            self.step_per_epoch = math.ceil(len(dataset) / cfg.train_batch_size)

        sampler = DistributedSampler(
            dataset,
            get_world_size(),
            get_rank(),
            shuffle=cfg.get(f'{mode}_data_shuffle', False)
        ) if torch.distributed.is_initialized() and mode == BasicTSMode.TRAIN else None

        shuffle = False if torch.distributed.is_initialized() and mode == BasicTSMode.TRAIN\
              else cfg.get(f'{mode}_data_shuffle', False)

        return (DataLoaderX if cfg.get(f'{mode}_data_prefetch') else DataLoader)(
            dataset,
            collate_fn=cfg.get(f'{mode}_data_collate_fn', None),
            batch_size=cfg.get(f'{mode}_batch_size', 1),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.get(f'{mode}_data_num_workers', 0),
            pin_memory=cfg.get(f'{mode}_data_pin_memory', False)
        )

    def _count_parameters(self) -> None:
        """Count parameters of the model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Total parameters: {total_params}')
        self.logger.info(f'Trainable parameters: {trainable_params}')

    @torch.no_grad()
    @master_only
    # pylint: disable=unused-argument
    def inference_pipeline(self, cfg: Optional[Dict] = None, input_data: Union[str, list] = '', output_data_file_path: str = '', **kwargs) -> tuple:
        """
        The complete inference process.

        Args:
            cfg (Dict, optional): Configuration dictionary.
            input_data (Union[str, list], optional): The input data file path or data list for inference.
            output_data_file_path (str, optional): The output data file path. Defaults to '' meaning no output file.
            **kwargs: Additional keyword arguments for flexibility in inference.
        """

        # if isinstance(input_data, str):
        #     pass

        self._init_inference(cfg, input_data)

        self.on_inference_start()

        inference_start_time = time.time()
        self.model.eval()

        # execute the inference process
        result = self.inference(save_result_path=output_data_file_path)

        inference_end_time = time.time()
        self.update_meter('inference/time', inference_end_time - inference_start_time)

        self.print_meters('inference')

        # logging here for intuitiveness
        if output_data_file_path:
            self.logger.info(f'inference results saved to {output_data_file_path}.')

        self.on_inference_end()

        return result

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

    def _save_metrics(self) -> None:
        """Save metrics results to self.ckpt_save_dir/test_metrics.json"""

        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        # for i in self.evaluation_horizons:
        #     metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}
        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)
        self.logger.info(f'Test metrics saved to {os.path.join(self.ckpt_save_dir, "test_metrics.json")}.')

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
            self.optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict[self.training_unit]
            if checkpoint_dict.get('best_metrics') is not None:
                self.best_metrics = checkpoint_dict['best_metrics']
            if self.lr_scheduler is not None:
                self.lr_scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('Resume training')
            if self.amp_scaler is not None:
                self.amp_scaler.load_state_dict(checkpoint_dict['amp_scaler_state_dict'])

        except (IndexError, OSError, KeyError):
            pass

    @master_only
    def _save_model(self, unit_count: int):
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
            self.training_unit: unit_count,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics
        }

        # save learning rate scheduler and amp scaler if available
        if self.lr_scheduler is not None:
            ckpt_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.amp_scaler is not None:
            ckpt_dict['amp_scaler_state_dict'] = self.amp_scaler.state_dict()

        # backup last epoch
        eqv_epoch = unit_count // self.val_interval if self.training_unit == 'step' else unit_count
        # eqv_save_strategy = self.ckpt_save_strategy TODO: step strategy should also be transformed
        last_ckpt_path = self._get_ckpt_path(eqv_epoch - 1)
        backup_last_ckpt(last_ckpt_path, eqv_epoch, self.ckpt_save_strategy)

        # save ckpt
        ckpt_path = self._get_ckpt_path(unit_count)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10 epoch or in the end
        if self.training_unit == 'epoch':
            if eqv_epoch % 10 == 0 or unit_count == self.num_epochs:
                clear_ckpt(self.ckpt_save_dir)
        else: # self.training_unit == 'step'
            if eqv_epoch % 10 == 0 or unit_count == self.num_steps:
                clear_ckpt(self.ckpt_save_dir)

    @master_only
    def _save_best_model(self, unit_count: int, metric_name: str, greater_best: bool = True):
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

        metric = self.meter_pool.get_value(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                self.training_unit: unit_count,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'best_metrics': self.best_metrics
            }
            # save learning rate scheduler and amp scaler if available
            if self.lr_scheduler is not None:
                ckpt_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
            if self.amp_scaler is not None:
                ckpt_dict['amp_scaler_state_dict'] = self.amp_scaler.state_dict()
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)

    @master_only
    def _save_results(self, batch_idx: int, batch_data: Dict[str, torch.Tensor]) -> None:

        """
        Save the test results to disk.

        Args:
            batch_idx (int): The index of the current batch.
            batch_data (Dict[np.ndarray]): The test results:{
                'inputs': np.ndarray,
                'prediction': np.ndarray,
                'target': np.ndarray,
            }
        """

        inputs = batch_data['inputs'].detach().cpu().numpy()
        prediction = batch_data['prediction'].detach().cpu().numpy()
        target = batch_data['target'].detach().cpu().numpy()

        total_samples = len(self.test_data_loader.dataset)

        save_dir = os.path.join(self.ckpt_save_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)
        inputs_path = os.path.join(save_dir, 'inputs.npy')
        pred_path = os.path.join(save_dir, 'prediction.npy')
        tgt_path = os.path.join(save_dir, 'target.npy')

        # create memmap files
        if batch_idx == 0:
            self._inputs_memmap = np.memmap(inputs_path, dtype=inputs.dtype, mode='w+',
                                    shape=(total_samples, *inputs.shape[1:]))
            self._prediction_memmap = np.memmap(pred_path, dtype=prediction.dtype, mode='w+',
                                    shape=(total_samples, *prediction.shape[1:]))
            self._target_memmap = np.memmap(tgt_path, dtype=target.dtype, mode='w+',
                                shape=(total_samples, *target.shape[1:]))

        start = batch_idx * inputs.shape[0]
        end = start + inputs.shape[0]

        self._inputs_memmap[start:end] = inputs
        self._prediction_memmap[start:end] = prediction
        self._target_memmap[start:end] = target

        if batch_idx == (total_samples // inputs.shape[0]):
            self._inputs_memmap.flush()
            self._prediction_memmap.flush()
            self._target_memmap.flush()

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

    def _get_ckpt_save_dir(self, cfg: 'BasicTSConfig') -> str:
        """Get checkpoint save dir.

        The format is "{exp_name}/{model_name}_{time}"

        Returns:
            checkpoint save dir (str)
        """

        return os.path.join(cfg.ckpt_save_dir, cfg.md5)

    def _get_ckpt_path(self, unit_count: int) -> str:
        """Get checkpoint path.

        The format is "{ckpt_save_dir}/{model_name}_{epoch}"

        Args:
            epoch (int): current epoch.

        Returns:
            checkpoint path (str)
        """

        unit_count_str = str(unit_count).zfill(len(str(self.num_steps if self.training_unit == 'step' else self.num_epochs)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, unit_count_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

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
    def register_meter(self, name, meter_type, fmt='{:f}', plt=True) -> None:
        self.meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_meter(self, name, value, n=1) -> None:
        self.meter_pool.update(name, value, n)

    @master_only
    def print_meters(self, meter_type) -> None:
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_meters(self, meter_type, step) -> None:
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer)

    @master_only
    def reset_meters(self) -> None:
        self.meter_pool.reset()

    # endregion meters and tensorboard
