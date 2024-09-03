from typing import Optional, Dict

import torch
from tqdm import tqdm
from easytorch.utils.dist import master_only
from basicts.runners import BaseTimeSeriesForecastingRunner


class TSFormerRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)

        # feed forward
        reconstruction_masked_tokens, label_masked_tokens = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
        results = {'prediction': reconstruction_masked_tokens, 'target': label_masked_tokens, 'inputs': history_data}
        return results

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:

        for data in tqdm(self.test_data_loader):
            data = self.preprocessing(data)
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
            # re-scale data
            forward_return = self.postprocessing(forward_return)
            # metrics
            if not self.if_evaluate_on_gpu:
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()

            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(forward_return['prediction'], forward_return['target'], null_val=self.null_val)
                self.update_epoch_meter("test_"+metric_name, metric_item.item())
