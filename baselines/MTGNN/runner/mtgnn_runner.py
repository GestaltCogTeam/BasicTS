from typing import Tuple, Union

import torch
import numpy as np

from basicts.runners import SimpleTimeSeriesForecastingRunner


class MTGNNRunner(SimpleTimeSeriesForecastingRunner):
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
        if train:
            future_data, history_data, idx = data['target'], data['inputs'], data['idx']
        else:
            future_data, history_data = data['target'], data['inputs']
            idx = None

        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, seq_len, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        # model forward
        model_return = self.model(history_data=history_data, idx=idx, batch_seen=iter_num, epoch=epoch)   # B, L, N, C

        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        model_return["inputs"] = self.select_target_features(history_data)
        model_return["target"] = self.select_target_features(future_data)
        assert list(model_return["prediction"].shape)[:3] == [batch_size, seq_len, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        return model_return

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:

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
            future_data, history_data = data['target'][:, :, idx, :], data['inputs'][:, :, idx, :]
            data = {
                'target': future_data,
                'inputs': history_data,
                'idx': idx
            }
            loss = super().train_iters(epoch, iter_index, data)
            self.backward(loss)
