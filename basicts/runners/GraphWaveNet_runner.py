import torch
from basicts.runners.base_traffic_runner import TrafficRunner

class GraphWaveNetRunner(TrafficRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def data_reshaper(self, data: torch.Tensor) -> torch.Tensor:
        """reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """
        # reshape data
        data = data.transpose(1, 3)
        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, self.forward_features, :, :]
        return data
    
    def data_i_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshape data back to the BasicTS framework

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

    def forward(self, data: tuple, iter_num: int = None, epoch:int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            iter_num (int, optional): iteration number. Defaults to None.
            epoch (int, optional): epoch number. Defaults to None.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """
        # preprocess
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        B, L, N, C      = history_data.shape
        
        history_data    = self.data_reshaper(history_data)

        # feed forward
        prediction_data = self.model(history_data=history_data, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [B, L, N], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.data_i_reshape(prediction_data)
        real_value = self.data_i_reshape(future_data)
        return prediction, real_value
