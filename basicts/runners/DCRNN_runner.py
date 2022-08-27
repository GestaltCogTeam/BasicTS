import torch
from basicts.runners.short_mts_runner import MTSRunner

class DCRNNRunner(MTSRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def setup_graph(self, data):
        try:
            self.train_iters(data, 0, 0)
        except AttributeError:
            pass

    def data_reshaper(self, data: torch.Tensor, channel=None) -> torch.Tensor:
        """select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]
            channel (list): self-defined selected channels
        Returns:
            torch.Tensor: reshaped data
        """
        # select feature using self.forward_features
        if self.forward_features is not None and channel is None:
            data = data[:, :, :, self.forward_features]
        if channel is not None:
            data = data[:, :, :, channel]
        # reshape data [B, L, N, C] -> [L, B, N*C] (DCRNN required)
        B, L, N, C = data.shape
        data = data.reshape(B, L, N*C)      # [B, L, N*C]
        data = data.transpose(0, 1)         # [L, B, N*C]
        return data
    
    def data_i_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """
        # reshape data
        pass
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
        B, L, N, C      = future_data.shape
        
        history_data    = self.data_reshaper(history_data)
        if train:
            future_data_    = self.data_reshaper(future_data, channel=[0])      # teacher forcing only use the first dimension.
        else:
            future_data_    = None

        # feed forward
        prediction_data = self.model(history_data=history_data, future_data=future_data_, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [B, L, N], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.data_i_reshape(prediction_data)
        real_value = self.data_i_reshape(future_data)
        return prediction, real_value
