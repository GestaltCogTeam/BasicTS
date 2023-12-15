import torch

from ..base_m4_runner import BaseM4Runner


class M4ForecastingRunner(BaseM4Runner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

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
            data (tuple): (future_data, history_data, future_mask, history_mask).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data, future_mask, history_mask = data
        history_data = self.to_running_device(history_data)      # B, L, 1, C
        future_data = self.to_running_device(future_data)       # B, L, 1, C
        history_mask = self.to_running_device(history_mask)     # B, L, 1
        future_mask = self.to_running_device(future_mask)       # B, L, 1

        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # model forward
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, history_mask=history_mask, future_mask=future_mask, batch_seen=iter_num, epoch=epoch, train=train)
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return * future_mask.unsqueeze(-1)}
        if "inputs" not in model_return: model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return: model_return["target"] = self.select_target_features(future_data * future_mask.unsqueeze(-1))
        
        return model_return
