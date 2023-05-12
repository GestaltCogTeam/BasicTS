import math
import functools
from typing import Tuple, Union, Optional

import torch
import numpy as np
from easydict import EasyDict
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import masked_mae, masked_mape, masked_rmse


class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for multivariate time series forecasting datasets.
    Features:
        - Evaluate at pre-defined horizons (1~12 as default) and overall.
        - Metrics: MAE, RMSE, MAPE. Allow customization. The best model is the one with the smallest mae at validation.
        - Support setup_graph for the models acting like tensorflow.
        - Loss: MAE (masked_mae) as default. Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg.get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg.get("DATASET_TYPE", " ")
        self.if_rescale = cfg.get("RESCALE", True)   # if rescale data when calculating loss or metrics, default as True

        # setup graph
        self.need_setup_graph = cfg["MODEL"].get("SETUP_GRAPH", False)

        # read scaler for re-normalization
        self.scaler = load_pkl("{0}/scaler_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]))
        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        # define metric
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg["TRAIN"].get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg["TRAIN"].CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg["TRAIN"].CL.get("CL_EPOCHS")
            self.prediction_length = cfg["TRAIN"].CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg["TRAIN"].CL.get("STEP_SIZE", 1)
        # evaluation
        self.if_evaluate_on_gpu = cfg.get("EVAL", EasyDict()).get("USE_GPU", True)     # evaluate on gpu or cpu (gpu is faster but may cause OOM)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get("EVAL", EasyDict()).get("HORIZONS", range(1, 13))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 1."

    def setup_graph(self, cfg: dict, train: bool):
        """Setup all parameters and the computation graph.
        Implementation of many works (e.g., DCRNN, GTS) acts like TensorFlow, which creates parameters in the first feedforward process.

        Args:
            cfg (dict): config
            train (bool): training or inferencing
        """

        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(enumerate(dataloader))[1] # get the first batch
        self.forward(data=data, epoch=1, iter_num=0, train=train)

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        # setup graph
        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False
        super().init_training(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False
        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.4f}")

    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

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

        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

    def rescale_data(self, data: torch.Tensor) -> torch.Tensor:
        """Rescale data.

        Args:
            data (torch.Tensor): data to be re-scaled.

        Returns:
            torch.Tensor: re-scaled data.
        """

        return SCALER_REGISTRY.get(self.scaler["func"])(data, **self.scaler["args"])

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """

        iter_num = (epoch-1) * self.iter_per_epoch + iter_index
        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        # re-scale data
        prediction = self.rescale_data(forward_return[0]) if self.if_rescale else forward_return[0]
        real_value = self.rescale_data(forward_return[1]) if self.if_rescale else forward_return[1]
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return[0] = prediction[:, :cl_length, :, :]
            forward_return[1] = real_value[:, :cl_length, :, :]
        else:
            forward_return[0] = prediction
            forward_return[1] = real_value
        loss = self.metric_forward(self.loss, forward_return)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction, real_value])
            self.update_epoch_meter("train_"+metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        # re-scale data
        prediction = self.rescale_data(forward_return[0]) if self.if_rescale else forward_return[0]
        real_value = self.rescale_data(forward_return[1]) if self.if_rescale else forward_return[1]
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction, real_value])
            self.update_epoch_meter("val_"+metric_name, metric_item.item())

    def evaluate(self, prediction, real_value):
        """Evaluate the model on test data.

        Args:
            prediction (torch.Tensor): prediction data [B, L, N, C].
            real_value (torch.Tensor): ground truth [B, L, N, C].
        """

        if not self.if_evaluate_on_gpu:
            prediction = prediction.detach().cpu()
            real_value = real_value.detach().cpu()
        # test performance of different horizon
        for i in self.evaluation_horizons:
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            # metrics
            metric_repr = ""
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_repr += ", Test {0}: {1:.4f}".format(metric_name, metric_item.item())
            log = "Evaluate best model on test data for horizon {:d}" + metric_repr
            log = log.format(i+1)
            self.logger.info(log)
        # test performance overall
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction, real_value])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())

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
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0])        # preds = forward_return[0]
            real_value.append(forward_return[1])        # testy = forward_return[1]
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = self.rescale_data(prediction) if self.if_rescale else prediction
        real_value = self.rescale_data(real_value) if self.if_rescale else real_value
        # evaluate
        self.evaluate(prediction, real_value)

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)
