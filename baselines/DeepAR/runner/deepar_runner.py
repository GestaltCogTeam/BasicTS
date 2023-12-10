from typing import Dict
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

    def rescale_data(self, input_data: Dict) -> Dict:
        """Rescale data.

        Args:
            data (Dict): Dict of data to be re-scaled.

        Returns:
            Dict: Dict re-scaled data.
        """

        if self.if_rescale:
            input_data["inputs"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["inputs"], **self.scaler["args"])
            input_data["prediction"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["prediction"], **self.scaler["args"])
            input_data["target"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["target"], **self.scaler["args"])
            if "mus" in input_data.keys():
                input_data["mus"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["mus"], **self.scaler["args"])
            if "sigmas" in input_data.keys():
                input_data["sigmas"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["sigmas"], **self.scaler["args"])
        return input_data

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction =[]
        target = []
        inputs = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            if not self.if_evaluate_on_gpu:
                forward_return["prediction"] = forward_return["prediction"].detach().cpu()
                forward_return["target"] = forward_return["target"].detach().cpu()
                forward_return["inputs"] = forward_return["inputs"].detach().cpu()
            prediction.append(forward_return["prediction"])
            target.append(forward_return["target"])
            inputs.append(forward_return["inputs"])
        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)
        # re-scale data
        returns_all = self.rescale_data({"prediction": prediction[:, -self.output_seq_len:, :, :], "target": target[:, -self.output_seq_len:, :, :], "inputs": inputs})
        # evaluate
        self.evaluate(returns_all)

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            dict: keys that must be included: inputs, prediction, target
        """

        # preprocess
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # model forward
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)

        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return:
            model_return["target"] = self.select_target_features(future_data)
        return model_return
