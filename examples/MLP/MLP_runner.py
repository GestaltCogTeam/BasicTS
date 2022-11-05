import torch

from basicts.runners import BaseTimeSeriesForecastingRunner


class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features.
        Copy from basicts.runners.simple_tsf_runner.py."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        _future_data = self.select_input_features(future_data)

        # curriculum learning
        if self.cl_param is None:
            prediction_data = self.model(
                history_data=history_data, future_data=_future_data, batch_seen=iter_num, epoch=epoch, train=train)
        else:
            task_level = self.curriculum_learning(epoch)
            prediction_data = self.model(history_data=history_data, future_data=_future_data,
                                         batch_seen=iter_num, epoch=epoch, train=train, task_level=task_level)
        # feed forward
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value
