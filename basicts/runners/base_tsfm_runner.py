# Time Series Foundation Model Runner


# TODO Tasks
# complie model

# Doing Task
# 更新添加scheduler
# lr scheduler需要每一个iteration都动
# valid set 应该是一个有结束的datasets（见valid函数，用的tqdm(valid_dataset)而不是range(self.eval_iters)

# DONE Tasks
# 将epoch改成iteration
# grad accumulation
# ddp
# amp


import os
import json
import inspect
import functools
from typing import Tuple, Union, Optional, Dict

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from easydict import EasyDict
from easytorch.device import _DEVICE_TYPE
from easytorch.utils import master_only, get_world_size

from .base_iteration_runner import BaseIterationRunner


class BaseTimeSeriesFoundationModelRunner(BaseIterationRunner):
    """
    Runner for multivariate time series forecasting tasks.
    
    Features:
        - supoort amp
        - support gradient accumulation
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # initialize scaler
        self.data_scaler = self.build_data_scaler(cfg)

        # define loss function
        self.loss = cfg['TRAIN']['LOSS']

        # define metrics
        self.metrics = cfg.get('METRICS', {}).get('FUNCS', {})
        self.target_metrics = cfg.get('METRICS', {}).get('TARGET', 'loss')
        self.metrics_best = cfg.get('METRICS', {}).get('BEST', 'min')
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'
        # handle null values in datasets, e.g., 0.0 or np.nan.
        self.null_val = cfg.get('METRICS', {}).get('NULL_VAL', np.nan)

        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)

        # test settings
        self.if_evaluate_on_gpu = cfg.get('EVAL', EasyDict()).get('USE_GPU', True)

        # automatic mixed precision (amp)
        assert _DEVICE_TYPE == 'gpu', 'AMP only supports CUDA.'
        self.model_dtype = cfg.get('MODEL.DTYPE', 'float32')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.model_dtype]
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        # GradScaler will scale up gradients and some of them might become inf, which may cause lr_scheduler throw incorrect warning information. See:
        # https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=(self.model_dtype == 'float16'), init_scale=2 ** 12) # init_scale

        # ddp with gradient accumulation
        self.grad_accumulation_steps = cfg.get('TRAIN.GRAD_ACCUMULATION_STEPS',1)

    def build_data_scaler(self, cfg: Dict):
        """Build scaler.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Scaler instance or None if no scaler is declared.
        """

        if 'SCALER' in cfg:
            return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_parameters}')

    def init_training(self, cfg: Dict):
        """Initialize training components, including loss, meters, etc.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_training(cfg)
        self.count_parameters()
        self.register_iteration_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_iteration_meter(f'train/{key}', 'train', '{:.4f}')

        ddp_world_size = get_world_size()
        grad_accumulation_steps = self.grad_accumulation_steps if self.grad_accumulation_steps else 1
        batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
        effective_batch_size = batch_size * ddp_world_size * grad_accumulation_steps
        device = _DEVICE_TYPE.upper() if ddp_world_size == 1 else _DEVICE_TYPE.upper() + 's'
        self.logger.info(f'Training with {ddp_world_size} {device}, batch size per {device}: {batch_size}, grad_accumulation_steps: {grad_accumulation_steps}')
        self.logger.info(f'Effective batch size: {effective_batch_size}')

    def init_validation(self, cfg: Dict):
        """Initialize validation components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_validation(cfg)
        self.register_iteration_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_iteration_meter(f'val/{key}', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        """Initialize test components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_test(cfg)
        self.register_iteration_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_iteration_meter(f'test/{key}', 'test', '{:.4f}')

    def build_train_dataset(self, cfg: Dict) -> Dataset:
        """Build the training dataset."""
        return self.build_dataset(cfg, mode='train')

    def build_val_dataset(self, cfg: Dict) -> Dataset:
        """Build the validation dataset."""
        return self.build_dataset(cfg, mode='valid')

    def build_test_dataset(self, cfg: Dict) -> Dataset:
        """Build the test dataset."""
        return self.build_dataset(cfg, mode='test')

    def build_dataset(self, cfg: Dict, mode:str) -> Dataset:
        """Build the training/validation/test dataset.

        Args:
            cfg (Dict): Configuration.
            mode (str): Mode.

        Returns:
            Dataset: The constructed dataset.
        """
        assert mode in ['train', 'valid', 'test']
        if 'DATASET' in cfg:
            # Training/validation/test share the same dataset with different mode.
            dataset = cfg['DATASET']['TYPE'](mode=mode, logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'{mode.capitalize()} dataset length: {len(dataset)}')
            assert True, 'debug here to see the length of infinite dataset'
        else:
            # Training/validation/test use different dataset.
            if 'logger' in inspect.signature(cfg[mode.upper()]['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg[mode.upper()]['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg[mode.upper()]['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg[mode.upper()]['DATA']['DATASET']['PARAM']['mode'] = mode
            dataset = cfg[mode.upper()]['DATA']['DATASET']['TYPE'](**cfg['TRAIN']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'{mode.capitalize()} dataset length: {len(dataset)}')
        return dataset

    def forward(self, data: tuple, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 
        Note: The outputs are not re-scaled.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        raise NotImplementedError()

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.data_scaler is not None:
            input_data['target'] = self.data_scaler.transform(input_data['target'])
            input_data['inputs'] = self.data_scaler.transform(input_data['inputs'])
        # TODO: add more preprocessing steps as needed.
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        # rescale data
        if self.data_scaler is not None and self.data_scaler.rescale:
            input_data['prediction'] = self.data_scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.data_scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.data_scaler.inverse_transform(input_data['inputs'])

        # subset forecasting
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data

    def train_iters(self, iteration: int, dataloader: DataLoader) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            iteration (int): current iteration.
            dataloader (torch.utils.data.DataLoader):dataloader.

        Returns:
            loss (torch.Tensor)
        """

        for micro_step in range(self.grad_accumulation_steps):
            # gradient accumulation
            accumulating = micro_step != (self.grad_accumulation_steps - 1)
            data = next(dataloader)
            data = self.preprocessing(data)
            with self.ctx:
                forward_return = self.forward(data=data, iter_num=iteration, train=True)
                forward_return = self.postprocessing(forward_return)
                loss = self.metric_forward(self.loss, forward_return)
                loss = loss / self.grad_accumulation_steps
            self.backward(loss, accumulating=accumulating)

        # update lr_scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.update_iteration_meter('train/loss', loss.item() * self.grad_accumulation_steps)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_iteration_meter(f'train/{metric_name}', metric_item.item())
        return loss

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:
        """Compute metrics using the given metric function.

        Args:
            metric_func (function or functools.partial): Metric function.
            args (Dict): Arguments for metrics computation.

        Returns:
            torch.Tensor: Computed metric value.
        """

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        if isinstance(metric_func, functools.partial):
            if 'null_val' not in metric_func.keywords and 'null_val' in covariate_names: # null_val is required but not provided
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if 'null_val' in covariate_names: # null_val is required
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item

    def backward(self, loss: torch.Tensor, accumulating: bool) -> None:
        """Backward and update params

        Args:
            loss (torch.Tensor): loss
            accumulating (bool): if in the gradient accumulation process

        """

        if accumulating and isinstance(self.model, DDP):
            with self.model.no_sync():
                self.amp_scaler.scale(loss).backward()
        if not accumulating:
            self.amp_scaler.scale(loss).backward()
            if self.clip_grad_param is not None:
                self.amp_scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
            self.amp_scaler.step(self.optim)
            self.amp_scaler.update()
            self.optim.zero_grad(set_to_none=True)

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        data = self.preprocessing(data)
        # TODO: consider using amp for validation
        # with self.ctx:
        forward_return = self.forward(data=data, iter_num=iter_index, train=False)
        forward_return = self.postprocessing(forward_return)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_iteration_meter('val/loss', loss)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_iteration_meter(f'val/{metric_name}', metric_item.item())

    def compute_evaluation_metrics(self, returns_all: Dict):
        """Compute metrics for evaluating model performance during the test process.

        Args:
            returns_all (Dict): Must contain keys: inputs, prediction, target.
        """

        metrics_results = {}

        metrics_results['overall'] = {}

        loss = self.metric_forward(self.loss, returns_all)
        self.update_iteration_meter('test/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_iteration_meter(f'test/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        return metrics_results

    @torch.no_grad()
    @master_only
    def test(self, train_iteration: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.
        
        Args:
            train_iteration (Optional[int]): Current iteration if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        prediction, target, inputs = [], [], []

        for data in self.test_data_loader:
            data = self.preprocessing(data)
            forward_return = self.forward(data, iter_num=None, train=False)
            forward_return = self.postprocessing(forward_return)

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        metrics_results = self.compute_evaluation_metrics(returns_all)

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/test_results.npz
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/test_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        return returns_all

    @master_only
    def on_validating_end(self, train_iteration: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_iteration (Optional[int]): Current iteration if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        if train_iteration is not None:
            self.save_best_model(train_iteration, 'val/' + self.target_metrics, greater_best=greater_best)
