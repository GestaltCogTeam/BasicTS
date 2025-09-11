import os
from collections import defaultdict
from typing import Dict, Tuple, Union

import numpy as np
import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class SelectiveTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """
    Runner for Time Series Selective Learning, where the function `train_iters` is overridden.

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

        # index_shape = (num_samples,)
        self.noise_mask_ratio = cfg['TRAIN'].get('NOISE_MASK_RATIO', None)
        self.anomaly_mask_ratio = cfg['TRAIN'].get('ANOMALY_MASK_RATIO', None)
        self.noise_mask = None
        self.current_loss = None
        self.random_select = cfg['TRAIN'].get('RANDOM_SELECT', False)
        
        # self.saved_target = None
        # self.saved_pred = None
        # self.saved_mask = None

        self.history_loss = self._get_history_loss_file(cfg)

    def _get_history_loss_file(self, cfg) -> np.ndarray:
        dataset_name = cfg['DATASET']['PARAM']['dataset_name']
        # input_len = cfg['DATASET']['PARAM']['input_len']
        input_len = 36 if dataset_name == 'Illness' else 336
        output_len = cfg['DATASET']['PARAM']['output_len']
        epoch = {
            'ETTh1': 100,
            'ETTh2': 50,
            'ETTm1': 10,
            'ETTm2': 10,
            'Electricity': 30,
            'Weather': 20,
            'Illness': 100,
            'ExchangeRate': 100,
            'Traffic': 50
        }
        loss_file_path = f'./history_loss/{dataset_name}_{input_len}_{output_len}/history_loss_{epoch[dataset_name]}.npy'
        return np.load(loss_file_path, allow_pickle=True)

    def on_train_start(self, cfg: Dict) -> None:
        
        super().on_train_start(cfg)
        
        num_samples = len(self.train_data_loader.dataset)
        loss_shape = (num_samples, self.output_len, self.num_nodes)
        self.current_loss = np.zeros(loss_shape, dtype=np.float32)
        # self.saved_mask = np.zeros(loss_shape, dtype=bool)
        # self.saved_target = np.zeros(loss_shape, dtype=np.float32)
        # self.saved_pred = np.zeros(loss_shape, dtype=np.float32)


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
        res_ = res.detach().cpu().numpy()
        self.current_loss[idx] = res_[..., 0]
    
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        if self.history_loss is not None :
                    #history_res = torch.abs(history_res)
            history_res = torch.tensor(np.abs(self.history_loss[idx]), device=res.device).unsqueeze(-1)
            excess_res = history_res - res
            # print(excess_res.mean().item())
            thresholds = torch.quantile(
                excess_res, 1 - self.anomaly_mask_ratio, dim=1, keepdim=True
            )
            
            forward_return['res_mask'] = excess_res > thresholds
            # res_mask = excess_res > thresholds
            # forward_return['history_res'] = self.history_loss[idx]
        
        if self.random_select:
            forward_return['res_mask'] = torch.rand_like(forward_return['res_mask'], dtype=float) < self.anomaly_mask_ratio

        #self.cov_mask = None
        if self.noise_mask is not None:
            expanded_idx = idx[:, np.newaxis] + np.arange(self.output_len) # [B, L]
            forward_return['cov_mask'] = self.noise_mask[expanded_idx].unsqueeze(-1)
        
        # if epoch == 10 or epoch == 20:
        #     self.saved_target[idx] = forward_return['target'][..., 0].detach().cpu().numpy()
        #     self.saved_pred[idx] = forward_return['prediction'][..., 0].detach().cpu().numpy()
        #     self.saved_mask[idx] = (self.cov_mask[expanded_idx] | forward_return['res_mask'][..., 0]).detach().cpu().numpy()

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

        # if epoch == 10 or epoch == 20:
        #     np.save(f'./samples/mask_ETTh1_336_{epoch}.npy', self.saved_mask)
        #     np.save(f'./samples/pred_ETTh1_336_{epoch}.npy', self.saved_pred)
        #     np.save(f'./samples/target_ETTh1_336_{epoch}.npy', self.saved_target)

        # if epoch ==1 or epoch == 20 or epoch == 50 or epoch == 100 and self.save_loss:
        #     np.save('./history_loss/ExchangeRate_336_336/history_loss_'+str(epoch)+'.npy', self.current_loss)
        
        #res = self.to_running_device(torch.tensor(self.current_loss))
        res = torch.tensor(self.current_loss)
        coe_var = self.compute_id_stats(res)
        thresholds = torch.quantile(
            coe_var, 1 - self.noise_mask_ratio, dim=0, keepdim=True
        )
        #res_mask = coe_var >= thresholds
        self.noise_mask = coe_var >= thresholds

        super().on_epoch_end(epoch)

    def compute_id_stats(self, residual: torch.Tensor):
        # tensor shape: N x H x C
        N, H, C = residual.shape
        
        # 生成对角线索引
        ids = (torch.arange(N, device=residual.device)[:, None] + 
            torch.arange(H, device=residual.device)[None, :])  # shape (N, H)
        
        # 展平并准备数据
        x_flat = residual.view(-1, C)  # [N*H, C]
        ids_flat = ids.view(-1, 1).expand(-1, C)  # [N*H, C]
        
        # 初始化结果张量
        result_shape = (N + H - 1, C)
        sum_per_id = torch.zeros(result_shape, dtype=residual.dtype, device=residual.device)
        sum_squared_per_id = torch.zeros_like(sum_per_id)
        
        # 计算总和和平方和
        sum_per_id.scatter_add_(0, ids_flat, x_flat)
        sum_squared_per_id.scatter_add_(0, ids_flat, (residual ** 2).view(-1, C))
        
        # 计算元素数目
        counts = torch.bincount(ids.view(-1), minlength=N+H-1).to(dtype=residual.dtype)
        counts = counts.unsqueeze(-1).expand(-1, C)
        
        # 计算均值和标准差
        mean = sum_per_id / counts
        std = torch.sqrt((sum_squared_per_id / counts) - mean.pow(2))

        #coe_var = std / (torch.abs(mean) + 1e-8)
        coe_var = std
        
        return coe_var

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
