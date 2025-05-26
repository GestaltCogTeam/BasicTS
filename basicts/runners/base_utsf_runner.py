import functools
import inspect
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from easytorch.device import _DEVICE_TYPE
from easytorch.utils import get_world_size, master_only
from torch.utils.data import DataLoader, Dataset

from .base_iteration_runner import BaseIterationRunner


class BaseUniversalTimeSeriesForecastingRunner(BaseIterationRunner):
    """
    Runner for universal time series forecasting models.
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
        self.model_dtype = cfg.get('MODEL.DTYPE', 'float32')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.model_dtype]
        self.user_amp = self.model_dtype in ['float16', 'bfloat16']
        if self.user_amp: assert _DEVICE_TYPE == 'gpu', 'AMP only supports CUDA.'
        self.amp_context = torch.amp.autocast(device_type='cuda', dtype=ptdtype, enabled=self.user_amp)
        # GradScaler will scale up gradients and some of them might become inf, which may cause lr_scheduler throw incorrect warning information. See:
        # https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
        self.amp_scaler = torch.amp.GradScaler(enabled=self.user_amp)
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
        self.register_iteration_meter('train/grad_norm', 'train', '{:.4f}')
        if self.user_amp:
            self.register_iteration_meter('train/amp_scale', 'train', '{:.4f}')
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

        raw_loss_sum = 0.0

        for micro_step in range(self.grad_accumulation_steps):
            # gradient accumulation
            accumulating = micro_step != (self.grad_accumulation_steps - 1)
            data = next(dataloader)
            data = self.preprocessing(data)
            with self.amp_context:
                forward_return = self.forward(data=data, iter_num=iteration, train=True)
                forward_return = self.postprocessing(forward_return)
                loss = self.metric_forward(self.loss, forward_return)
                raw_loss_sum += loss.item()
                loss = loss / self.grad_accumulation_steps
            self.backward(loss, accumulating=accumulating)

        # update lr_scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.update_iteration_meter('train/loss', raw_loss_sum / self.grad_accumulation_steps)
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

        if accumulating:
            if hasattr(self.model, 'no_sync'):
                with self.model.no_sync():
                    self.amp_scaler.scale(loss).backward()
            else:
                self.amp_scaler.scale(loss).backward()
        else:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optim)
            grad_norm = sum(
                        param.grad.data.norm(2).item() ** 2 for param in self.model.parameters() if param.grad is not None
                    ) ** 0.5
            if self.clip_grad_param is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)

            self.update_iteration_meter('train/grad_norm', grad_norm)
            if self.user_amp:
                self.update_iteration_meter('train/amp_scale', self.amp_scaler.get_scale())

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

    @master_only
    def on_validating_end(self, train_iteration: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_iteration (Optional[int]): Current iteration if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        if train_iteration is not None:
            self.save_best_model(train_iteration, 'val/' + self.target_metrics, greater_best=greater_best)
