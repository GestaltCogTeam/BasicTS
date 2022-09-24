import time
import torch

from easytorch.utils.dist import master_only
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner


class TSFormerRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FROWARD_FEATURES", None)
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

        # preprocess
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        # feed forward
        reconstruction_masked_tokens, label_masked_tokens = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
        # assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            # "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        # prediction = self.select_target_features(prediction_data)
        # real_value = self.select_target_features(future_data)
        return reconstruction_masked_tokens, label_masked_tokens

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction_rescaled, real_value_rescaled, null_val=self.null_val)
            self.update_epoch_meter("test_"+metric_name, metric_item.item())
