import torch
import numpy as np
from tqdm import tqdm
from basicts.runners.base_traffic_runner import TrafficRunner
from basicts.utils.registry import SCALER_REGISTRY

class MTGNNRunner(TrafficRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.step_size = cfg.TRAIN.CUSTOM.STEP_SIZE
        self.num_nodes = cfg.TRAIN.CUSTOM.NUM_NODES
        self.num_split = cfg.TRAIN.CUSTOM.NUM_SPLIT

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """select input features.

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
        """select target feature

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
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """
        if train:
            future_data, history_data, idx = data
        else:
            future_data, history_data = data
            idx = None

        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        B, L, N, C      = history_data.shape

        history_data    = self.select_input_features(history_data)
        
        prediction_data = self.model(history_data=history_data, idx=idx, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [B, L, N], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

    def train_data_loop(self, data_iter: tqdm, epoch: int):
        """train data loop

        Args:
            data_iter (tqdm.std.tqdm): data iterator
            epoch (int): epoch number
        """
        for iter_index, data in enumerate(data_iter):
            if iter_index%self.step_size==0:
                perm = np.random.permutation(range(self.num_nodes))
            num_sub = int(self.num_nodes/self.num_split)
            for j in range(self.num_split):
                if j != self.num_split-1:
                    idx = perm[j * num_sub:(j + 1) * num_sub]
                    raise
                else:
                    idx = perm[j * num_sub:]
                idx  = torch.tensor(idx)
                future_data, history_data = data
                data = future_data[:, :, idx, :], history_data[:, :, idx, :], idx
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
