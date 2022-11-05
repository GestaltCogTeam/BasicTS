from typing import Tuple, Union

import torch
import numpy as np

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class MTGNNRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        # graph training
        self.step_size = cfg.TRAIN.CUSTOM.STEP_SIZE
        self.num_nodes = cfg.TRAIN.CUSTOM.NUM_NODES
        self.num_split = cfg.TRAIN.CUSTOM.NUM_SPLIT
        self.perm = None

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
        """Select target feature

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
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        if train:
            future_data, history_data, idx = data
        else:
            future_data, history_data = data
            idx = None

        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, seq_len, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        prediction_data = self.model(
            history_data=history_data, idx=idx, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [
            batch_size, seq_len, num_nodes], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

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

        if iter_index % self.step_size == 0:
            self.perm = np.random.permutation(range(self.num_nodes))
        num_sub = int(self.num_nodes/self.num_split)
        for j in range(self.num_split):
            if j != self.num_split-1:
                idx = self.perm[j * num_sub:(j + 1) * num_sub]
                raise
            else:
                idx = self.perm[j * num_sub:]
            idx = torch.tensor(idx)
            future_data, history_data = data
            data = future_data[:, :, idx, :], history_data[:, :, idx, :], idx
            loss = super().train_iters(epoch, iter_index, data)
            self.backward(loss)
