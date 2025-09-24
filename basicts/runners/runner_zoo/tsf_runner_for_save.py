import os
from collections import defaultdict
from typing import Dict, Tuple, Union

import numpy as np
import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class TimeSeriesForecastingRunnerForSave(BaseTimeSeriesForecastingRunner):
    """
    Selective Runner for Time Series Forecasting, where the function `train_iters` is overridden.
    Selects forward and target features. This runner is designed to handle most cases.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)
        # self.topk_percentage = cfg['TRAIN'].get('TOPK_PERCENTAGE', None)

        self.select_period = cfg['TRAIN'].get('SELECT_PERIOD', None)

        # loss_file_path = cfg['TRAIN'].get('LOSS_FILE_PATH', None)
        # num_samples = len(self.train_data_loader)
        self.output_len = cfg['DATASET']['PARAM']['output_len']
        self.num_nodes = cfg['DATASET'].get('NUM_NODES', None)
        self.dataset_name = cfg['DATASET']['PARAM']['dataset_name']
        # self.input_len = cfg['DATASET']['PARAM']['input_len']
        self.FINAL_EPOCH = cfg['TRAIN']['NUM_EPOCHS']

        # index_shape = (num_samples,)

        self.current_loss = None
        # self.current_loss_index = np.zeros(index_shape, dtype=np.int64)

        # define a dictionary to store the result of each epoch
        # self.residual_register = None
        # self.selective_weights = None


    def on_train_start(self, cfg: Dict) -> None:
        
        super().on_train_start(cfg)
        
        num_samples = len(self.train_data_loader.dataset)
        loss_shape = (num_samples, self.output_len, self.num_nodes)
        self.current_loss = np.zeros(loss_shape, dtype=np.float32)


    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.step_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        res: torch.Tensor = torch.abs(forward_return['prediction'] - forward_return['target'])
        idx = data['idx'].detach().numpy()
        
        #update the loss of each batch
        # if (epoch - 1) % self.select_period == 0:
        if epoch ==1 or epoch == self.FINAL_EPOCH:
            print(epoch)
            res = res.detach().cpu().numpy()
            self.current_loss[idx] = res[..., 0]
    
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        loss = self.metric_forward(self.loss, forward_return)
        self.update_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_meter(f'train/{metric_name}', metric_item.item())
        return loss

    
    def on_epoch_end(self, epoch: int) -> None:
        """
        Callback at the end of each epoch to handle validation and testing.

        Args:
            epoch (int): The current epoch number.
        """

        if epoch == self.FINAL_EPOCH:
        # if epoch == 1 or epoch == 30 or epoch == 40 or epoch == 50:
            input_len = 36 if self.dataset_name == "Illness" else 336
            np.save(f'./history_loss/{self.dataset_name}_{input_len}_{self.output_len}/{self.model_name}_'+str(epoch)+'.npy', self.current_loss)

        # if (epoch - 1) % self.select_period == self.select_period - 1:
        #     self.history_loss = self.current_loss
        #     self.current_loss.fill(0)

        super().on_epoch_end(epoch)

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
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
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        # subset forecasting
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

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

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        
        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, 
                                  batch_seen=iter_num, epoch=epoch, train=train)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected features with shape [B, L, N, C2].
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C2].
        """

        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        """
        Select target time series based on the target time series specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N1, C].

        Returns:
            torch.Tensor: Data with selected target time series and shape [B, L, N2, C].
        """

        data = data[:, :, self.target_time_series, :]
        return data
