from functools import wraps
import functools
import inspect
import sched
import lightning.pytorch as pl
from typing import Any, Callable, Dict, Optional, Union, List

import numpy as np
import torch

from basicts.metrics import ALL_METRICS, masked_mae
from basicts.scaler import BaseScaler


class BasicTimeSeriesForecastingModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        history_len: int,
        horizon_len: int,
        metrics: Optional[List[str]] = None,
        forward_features: Optional[List[int]] = None,
        target_features: Optional[List[int]] = None,
        target_time_series: Optional[List[int]] = None,
        scaler: Any = None,
        null_val: Any = np.nan,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.forward_features = forward_features
        self.target_features = target_features
        self.target_time_series = target_time_series
        self.scaler = scaler
        self.null_val = null_val

        self.metric_func_dict = self.init_metrics(metrics)
        if "loss" not in self.metric_func_dict:
            if hasattr(self, "loss_func"):
                self.metric_func_dict["loss"] = self.loss_func
            else:
                # self.logger.info('No loss function is provided. Using masked_mae as default.')
                self.metric_func_dict["loss"] = masked_mae

    def init_metrics(self, metrics: Optional[List[str]]) -> Dict[str, Callable]:
        if metrics is None:
            return ALL_METRICS
        return {name: ALL_METRICS[name] for name in metrics}

    def basicts_forward(self, data: Dict, **kwargs) -> Dict:
        """
        The forward function of original runner.

        Performs the forward pass for training, validation, and testing.

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data["target"], data["inputs"]
        # history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        # future_data = self.to_running_device(future_data)  # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        train = self.trainer.training
        # epoch = self.trainer.current_epoch
        # batch_seen = self.trainer.global_step

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self(
            history_data=history_data,
            future_data=future_data_4_dec,
            # batch_seen=batch_seen,
            # epoch=epoch,
            # train=train,
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {"prediction": model_return}
        if "inputs" not in model_return:
            model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return:
            model_return["target"] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return["prediction"].shape)[:3] == [
            batch_size,
            length,
            num_nodes,
        ], "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:
        """Compute metrics using the given metric function.

        Args:
            metric_func (function or functools.partial): Metric function.
            args (Dict): Arguments for metrics computation.

        Returns:
            torch.Tensor: Computed metric value.
        """

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        if isinstance(metric_func, functools.partial):
            if 'null_val' not in metric_func.keywords and 'null_val' in covariate_names: # null_val is required but not provided
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if 'null_val' in covariate_names: # null_val is required
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item
    
    
    def training_step(self, batch, batch_idx):
        forward_return = self.basicts_forward(batch)
        metrics = {}
        for metric_name, metric_func in self.metric_func_dict.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            metrics[f"train/{metric_name}"] = metric_item
        self.log_dict(metrics, on_step=True)
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        forward_return = self.basicts_forward(batch)
        metrics = {}
        for metric_name, metric_func in self.metric_func_dict.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            metrics[f"val/{metric_name}"] = metric_item
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics["val/loss"]

    def test_step(self, batch, batch_idx):
        forward_return = self.basicts_forward(batch)
        metrics = {}
        for metric_name, metric_func in self.metric_func_dict.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            metrics[f"test/{metric_name}"] = metric_item
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics["test/loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], [scheduler]

    def preprocessing(self, input_data: Dict, scale_keys=["target", "inputs"]) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.
            scale_keys (Optional[Union[List[str], str]], optional): 'all' means scale all, None means ingore all, it also can be specificated by a list of str. Defaults to None.

        Returns:
            Dict: Processed data.
        """
        if scale_keys is None:
            scale_keys = []
        elif scale_keys == "all":
            scale_keys = input_data.keys()

        if self.scaler is not None:
            for k in scale_keys:
                input_data[k] = self.scaler.transform(input_data[k])
        # TODO: add more preprocessing steps as needed.
        return input_data

    def postprocessing(
        self, input_data: Dict, scale_keys=["target", "inputs", "prediction"]
    ) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.
            scale_keys (Optional[Union[List[str], str]], optional): 'all' means scale all, None means ingore all, it also can be specificated by a list of str. Defaults to None.

        Returns:
            Dict: Processed data.
        """

        # rescale data
        if self.scaler is not None and self.scaler.rescale:
            if scale_keys is None:
                scale_keys = []
            elif scale_keys == "all":
                scale_keys = input_data.keys()
            for k in scale_keys:
                input_data[k] = self.scaler.inverse_transform(input_data[k])

        # subset forecasting
        if self.target_time_series is not None:
            input_data["target"] = input_data["target"][
                :, :, self.target_time_series, :
            ]
            input_data["prediction"] = input_data["prediction"][
                :, :, self.target_time_series, :
            ]

        # TODO: add more postprocessing steps as needed.
        return input_data

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected features with shape [B, L, N, C2].
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C2].
        """

        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        """
        Select target time series based on the target time series specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N1, C].

        Returns:
            torch.Tensor: Data with selected target time series and shape [B, L, N2, C].
        """

        data = data[:, :, self.target_time_series, :]
        return data
