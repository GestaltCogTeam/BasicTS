from typing import List
import torch
from basicts.data.registry import SCALER_REGISTRY
from easytorch.utils.dist import master_only

from basicts.runners import BaseTimeSeriesForecastingRunner


class DeepARRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.output_seq_len = cfg["DATASET_OUTPUT_LEN"]

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

    def rescale_data(self, input_data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Rescale data.

        Args:
            data (List[torch.Tensor]): list of data to be re-scaled.

        Returns:
            List[torch.Tensor]: list of re-scaled data.
        """
        prediction, real_value, mus, sigmas = input_data
        if self.if_rescale:
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])
            mus = SCALER_REGISTRY.get(self.scaler["func"])(mus, **self.scaler["args"])
            sigmas = SCALER_REGISTRY.get(self.scaler["func"])(sigmas, **self.scaler["args"])
        return [prediction, real_value, mus, sigmas]

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = list(self.forward(data, epoch=None, iter_num=None, train=False))
            if not self.if_evaluate_on_gpu:
                forward_return[0], forward_return[1] = forward_return[0].detach().cpu(), forward_return[1].detach().cpu()
            prediction.append(forward_return[0])        # preds = forward_return[0]
            real_value.append(forward_return[1])        # testy = forward_return[1]
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        if self.if_rescale:
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])[:, -self.output_seq_len:, :, :]
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])[:, -self.output_seq_len:, :, :]
        # evaluate
        self.evaluate(prediction, real_value)

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
        future_data_4_dec = self.select_input_features(future_data)

        # feed forward
        pred_values, real_values, mus, sigmas = self.model(history_data=history_data, future_data=future_data_4_dec, train=train)
        # post process
        prediction = self.select_target_features(pred_values)
        real_value = self.select_target_features(real_values)
        return prediction, real_value, mus, sigmas
