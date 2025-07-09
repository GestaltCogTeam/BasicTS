import functools
import inspect
import json
import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from easytorch.utils import master_only
from tqdm import tqdm

from basicts.data.simple_inference_dataset import TimeSeriesInferenceDataset

from ..metrics import (masked_mae, masked_mape, masked_mse, masked_rmse,
                       masked_wape)
from .base_epoch_runner import BaseEpochRunner


class BaseTimeSeriesForecastingRunner(BaseEpochRunner):
    """
    Runner for multivariate time series forecasting tasks.

    Features:
        - Supports evaluation at pre-defined horizons (optional) and overall performance assessment.
        - Metrics: MAE, RMSE, MAPE, WAPE, and MSE. Customizable. The best model is selected based on the smallest MAE on the validation set.
        - Supports `setup_graph` for models that operate similarly to TensorFlow.
        - Default loss function is MAE (masked_mae), but it can be customized.
        - Supports curriculum learning.
        - Users only need to implement the `forward` function.

    Customization:
        - Model:
            - Args:
                - history_data (torch.Tensor): Historical data with shape [B, L, N, C], 
                  where B is the batch size, L is the sequence length, N is the number of nodes, 
                  and C is the number of features.
                - future_data (torch.Tensor or None): Future data with shape [B, L, N, C]. 
                  Can be None if there is no future data available.
                - batch_seen (int): The number of batches seen so far.
                - epoch (int): The current epoch number.
                - train (bool): Indicates whether the model is in training mode.
            - Return:
                - Dict or torch.Tensor:
                    - If returning a Dict, it must contain the 'prediction' key. Other keys are optional and will be passed to the loss and metric functions.
                    - If returning a torch.Tensor, it should represent the model's predictions, with shape [B, L, N, C].

        - Loss & Metrics (optional):
            - Args:
                - prediction (torch.Tensor): Model's predictions, with shape [B, L, N, C].
                - target (torch.Tensor): Ground truth data, with shape [B, L, N, C].
                - null_val (float): The value representing missing data in the dataset.
                - Other args (optional): Additional arguments will be matched with keys in the model's return dictionary, if applicable.
            - Return:
                - torch.Tensor: The computed loss or metric value.

        - Dataset (optional):
            - Return: The returned data will be passed to the `forward` function as the `data` argument.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # setup graph flag
        self.need_setup_graph = cfg['MODEL'].get('SETUP_GRAPH', False)

        # initialize scaler
        self.scaler = self.build_scaler(cfg)

        # define loss function
        self.loss = cfg['TRAIN']['LOSS']

        # define metrics
        self.metrics = cfg.get('METRICS', {}).get('FUNCS', {
                                                            'MAE': masked_mae, 
                                                            'RMSE': masked_rmse, 
                                                            'MAPE': masked_mape, 
                                                            'WAPE': masked_wape, 
                                                            'MSE': masked_mse
                                                        })
        self.target_metrics = cfg.get('METRICS', {}).get('TARGET', 'loss')
        self.metrics_best = cfg.get('METRICS', {}).get('BEST', 'min')
        assert self.target_metrics in self.metrics or self.target_metrics == 'loss', f'Target metric {self.target_metrics} not found in metrics.'
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'
        # handle null values in datasets, e.g., 0.0 or np.nan.
        self.null_val = cfg.get('METRICS', {}).get('NULL_VAL', np.nan)

        # curriculum learning setup
        self.cl_param = cfg['TRAIN'].get('CL', None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg['TRAIN'].CL.get('WARM_EPOCHS', 0)
            self.cl_epochs = cfg['TRAIN'].CL.get('CL_EPOCHS')
            self.prediction_length = cfg['TRAIN'].CL.get('PREDICTION_LENGTH')
            self.cl_step_size = cfg['TRAIN'].CL.get('STEP_SIZE', 1)

        # Eealuation settings
        self.if_evaluate_on_gpu = cfg.get('EVAL', EasyDict()).get('USE_GPU', True)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get('EVAL', EasyDict()).get('HORIZONS', [])]
        assert len(self.evaluation_horizons) == 0 or min(self.evaluation_horizons) >= 0, 'The horizon should start counting from 1.'

    def build_scaler(self, cfg: Dict):
        """Build scaler.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Scaler instance or None if no scaler is declared.
        """

        if 'SCALER' in cfg:
            return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def setup_graph(self, cfg: Dict, train: bool):
        """Setup all parameters and the computation graph.

        Some models (e.g., DCRNN, GTS) require creating parameters during the first forward pass, similar to TensorFlow.

        Args:
            cfg (Dict): Configuration.
            train (bool): Whether the setup is for training or inference.
        """

        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(iter(dataloader))  # get the first batch
        self.forward(data=data, epoch=1, iter_num=0, train=train)

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_parameters}')

    def init_training(self, cfg: Dict):
        """Initialize training components, including loss, meters, etc.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False

        super().init_training(cfg)
        self.count_parameters()

        self.register_epoch_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'train/{key}', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        """Initialize validation components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        super().init_validation(cfg)
        self.register_epoch_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'val/{key}', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):
        """Initialize test components, including meters.

        Args:
            cfg (Dict): Configuration.
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False

        super().init_test(cfg)
        self.register_epoch_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'test/{key}', 'test', '{:.4f}')

    def build_train_dataset(self, cfg: Dict):
        """Build the training dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed training dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['mode'] = 'train'
            dataset = cfg['TRAIN']['DATA']['DATASET']['TYPE'](**cfg['TRAIN']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        else:
            dataset = cfg['DATASET']['TYPE'](mode='train', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    def build_val_dataset(self, cfg: Dict):
        """Build the validation dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed validation dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['mode'] = 'valid'
            dataset = cfg['VAL']['DATA']['DATASET']['TYPE'](**cfg['VAL']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='valid', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')

        return dataset

    def build_test_dataset(self, cfg: Dict):
        """Build the test dataset.

        Args:
            cfg (Dict): Configuration.

        Returns:
            Dataset: The constructed test dataset.
        """

        if 'DATASET' not in cfg:
            # TODO: support building different datasets for training, validation, and test.
            if 'logger' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['mode'] = 'test'
            dataset = cfg['TEST']['DATA']['DATASET']['TYPE'](**cfg['TEST']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='test', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')

        return dataset

    def build_inference_dataset(self, cfg: Dict, input_data: Union[str, list]):
        """Build the inference dataset.

        Args:
            cfg (Dict): Configuration.
            input_data (Union[str, list]): The input data file path or data list for inference.
        Returns:
            Dataset: The constructed inference dataset.
        """

        dataset = TimeSeriesInferenceDataset(dataset=input_data, logger=self.logger, **cfg['DATASET']['PARAM'])
        self.logger.info(f'Inference dataset length: {len(dataset)}')

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level for curriculum learning.

        Args:
            epoch (int, optional): Current epoch if in training process; None otherwise. Defaults to None.

        Returns:
            int: Task level for the current epoch.
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still in warm-up phase
            cl_length = self.prediction_length
        else:
            progress = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(progress, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 
        Note: The outputs are not re-scaled.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
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

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('val/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item())

    def compute_evaluation_metrics(self, returns_all: Dict):
        """Compute metrics for evaluating model performance during the test process.

        Args:
            returns_all (Dict): Must contain keys: inputs, prediction, target.
        """

        metrics_results = {}
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue # MASE needs to be calculated after all horizons
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', Test {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate best model on test data for horizon {i + 1}{metric_repr}')

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_epoch_meter(f'test/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        return metrics_results

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.
        
        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        prediction, target, inputs = [], [], []

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('test/loss', loss.item())

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

    @torch.no_grad()
    @master_only
    def inference(self, save_result_path: str = '') -> tuple:
        """Inference process.
        
        Args:
            save_result_path (str): The path to save the inference results. Defaults to '' meaning no saving.
        """

        data = next(iter(self.inference_dataset_loader))

        forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
        prediction = forward_return['prediction'].detach().cpu()
        prediction = prediction.squeeze(0).squeeze(-1)

        datetime_data = self._inference_get_data_time(prediction, self.inference_dataset.last_datetime, \
                                                   self.inference_dataset.description['frequency (minutes)'])

        # save
        if save_result_path:
            # save prediction to save_result_path with csv format
            datetime_data = np.fromiter((x.astype('datetime64[us]').item().strftime('%Y-%m-%d %H:%M:%S')\
                                         for x in datetime_data), 'S32').reshape(-1, 1)
            save_data = np.concatenate((datetime_data, prediction.numpy().astype(str)), axis=1)
            np.savetxt(save_result_path, save_data, delimiter=',', fmt='%s')

        return prediction.numpy(), datetime_data

    @torch.no_grad()
    @master_only
    def _inference_get_data_time(self, data: np.ndarray, last_datatime: np.datetime64, freq: int) -> np.ndarray:
        """
        Append new data to the existing data
        """

        datetime_data = np.arange(last_datatime + np.timedelta64(freq, 'm'),
                             last_datatime + np.timedelta64(freq * (len(data) + 1), 'm'),
                             np.timedelta64(freq, 'm'))

        return datetime_data

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of the validation process.

        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
        """
        greater_best = not self.metrics_best == 'min'
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/' + self.target_metrics, greater_best=greater_best)
