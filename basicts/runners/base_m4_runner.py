import math
import inspect
import functools
from typing import Tuple, Union, Dict

import torch
import numpy as np
from easydict import EasyDict
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY


class BaseM4Runner(BaseRunner):
    """
    Runner for M4 dataset.
        - There is no validation set.
        - On training end, we inference on the test set and save the prediction results.
        - No metrics (but the loss). Since the evaluation is not done in this runner, thus no metrics are needed.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        assert "M4" in self.dataset_name, "M4Runner only supports M4 dataset."
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg.get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg.get("DATASET_TYPE", " ")
        self.if_rescale = None # no normalization in M4 dataset, so no need to rescale

        # setup graph
        self.need_setup_graph = cfg["MODEL"].get("SETUP_GRAPH", False)

        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        # define metric
        self.metrics = cfg.get("METRICS", {"loss": self.loss})
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
        assert len(self.evaluation_horizons) == 0 or min(self.evaluation_horizons) >= 0, "The horizon should start counting from 1."
        self.save_path = cfg.get("EVAL", EasyDict()).get("SAVE_PATH") # save path for inference results, should not be None

    def build_train_dataset(self, cfg: dict):
        """Build train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        data_file_path = "{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))
        index_file_path = "{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))
        mask_file_path = "{0}/mask_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mask_file_path"] = mask_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        data_file_path = "{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))
        index_file_path = "{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))
        mask_file_path = "{0}/mask_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", None))

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mask_file_path"] = mask_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)

        return dataset

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """
        raise NotImplementedError()

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

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Number of parameters: {0}".format(num_parameters))

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
        # init training
        super().init_training(cfg)
        # count parameters
        self.count_parameters()
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.6f}")

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
            self.register_epoch_meter("test_"+key, "test", "{:.6f}")

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

    def metric_forward(self, metric_func, args) -> torch.Tensor:
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (Dict): arguments for metrics computation.

        Returns:
            torch.Tensor: metric value.
        """
        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        if isinstance(metric_func, functools.partial):
            # support partial function
            # users can define their partial function in the config file
            # e.g., functools.partial(masked_mase, freq="4", null_val=np.nan)
            if "null_val" in covariate_names and "null_val" not in metric_func.keywords: # if null_val is required but not provided
                args["null_val"] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            # is a function
            # filter out keys that are not in function arguments
            metric_item = metric_func(**args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

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
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        # re-scale data
        forward_return = self.rescale_data(forward_return)
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return["prediction"] = forward_return["prediction"][:, :cl_length, :, :]
            forward_return["target"] = forward_return["target"][:, :cl_length, :, :]
        loss = self.metric_forward(self.loss, forward_return)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter("train_"+metric_name, metric_item.item())
        return loss

    def save_prediction(self, returns_all):
        """Evaluate the model on test data.

        Args:
            returns_all (Dict): must contain keys: inputs, prediction, target
        """
        prediction = returns_all["prediction"].detach().cpu().numpy()
        loss = self.metric_forward(self.loss, returns_all)
        self.update_epoch_meter("test_loss", loss.item())
        # save prediction as self.save_path/self.dataset_name.npy
        np.save("{0}/{1}.npy".format(self.save_path, self.dataset_name), prediction)

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # TODO: fix OOM: especially when inputs, targets, and predictions are saved at the same time.
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
        returns_all = self.rescale_data({"prediction": prediction, "target": target, "inputs": inputs})
        # evaluate
        self.save_prediction(returns_all)

    def rescale_data(self, input_data: Dict) -> Dict:
        """Rescale data.

        Args:
            data (Dict): Dict of data to be re-scaled.

        Returns:
            Dict: Dict re-scaled data.
        """

        if self.if_rescale:
            input_data["prediction"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["prediction"], **self.scaler["args"])
            input_data["target"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["target"], **self.scaler["args"])
            input_data["inputs"] = SCALER_REGISTRY.get(self.scaler["func"])(input_data["inputs"], **self.scaler["args"])
        return input_data
