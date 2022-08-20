import math
from typing import Tuple, Union, Optional
import torch
from torch import nn
import numpy as np
from basicts.runners.base_runner import BaseRunner
from basicts.utils.registry import SCALER_REGISTRY
from basicts.utils.serialization import load_pkl
from easytorch.utils.dist import master_only

class MTSRunner(BaseRunner):
    """Runner for short term multivariate time series forecasting datasets. Typically, models predict the future 12 time steps based on historical time series.
    Features:
        - Evaluate at horizon 3, 6, 12, and overall.
        - Metrics: MAE, RMSE, MAPE.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    
    Args:
        BaseRunner (easytorch.easytorch.runner): base runner
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg['DATASET_NAME']
        self.null_val = cfg['TRAIN'].get('NULL_VAL', 0)        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.dataset_type = cfg['DATASET_TYPE']
        self.forward_features = cfg['MODEL'].get('FROWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)

        # read scaler for re-normalization
        self.scaler = load_pkl("datasets/" + self.dataset_name + "/scaler.pkl")
        # define loss
        self.loss = cfg['TRAIN']['LOSS']
        # define metric
        self.metrics = cfg['METRICS']        
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg.TRAIN.get('CL', None)
        if self.cl_param is not None:
            self.warm_up_epochs     = cfg.TRAIN.CL.get('WARM_EPOCHS', 0)
            self.cl_epochs          = cfg.TRAIN.CL.get('CL_EPOCHS')
            self.prediction_length  = cfg.TRAIN.CL.get('PREDICTION_LENGTH')

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """
        
        super().init_training(cfg)
        for key, value in self.metrics.items():
            self.register_epoch_meter("train_"+key, 'train', '{:.4f}')

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, value in self.metrics.items():
            self.register_epoch_meter("val_"+key, 'val', '{:.4f}')

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)
        for key, value in self.metrics.items():
            self.register_epoch_meter("test_"+key, 'test', '{:.4f}')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        """Define model.

        If you have multiple models, insert the name and class into the dict below,
        and select it through ```config```.

        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """

        return cfg['MODEL']['ARCH'](**cfg.MODEL.PARAM)

    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/index.pkl"
        batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
        dataset = cfg['DATASET_CLS'](raw_file_path, index_file_path, mode='train')
        print("train len: {0}".format(len(dataset)))
        
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/index.pkl"
        dataset = cfg['DATASET_CLS'](raw_file_path, index_file_path, mode='valid')
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

        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/index.pkl"
        dataset = cfg['DATASET_CLS'](raw_file_path, index_file_path, mode='test')
        print("test len: {0}".format(len(dataset)))
        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """calculate task level in curriculum learning.

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
            _ = (epoch - self.warm_up_epochs) // self.cl_epochs + 1
            cl_length = min(_, self.prediction_length)
        return cl_length

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
        raise NotImplementedError()

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
        prediction, real_value = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            loss = self.loss(prediction[:, :cl_length, :, :], real_value[:, :cl_length, :, :], null_val=self.null_val)
        else:
            loss = self.loss(prediction, real_value, null_val=self.null_val)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, null_val=self.null_val)
            self.update_epoch_meter('train_'+metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """

        prediction, real_value = self.forward(data=data, epoch=None, iter_num=None, train=False)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # loss
        mae  = self.loss(prediction, real_value, null_val=self.null_val)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, null_val=self.null_val)
            self.update_epoch_meter('val_'+metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: int = None):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction = []
        real_value  = []
        for iter_index, data in enumerate(self.test_data_loader):
            preds, testy = self.forward(data, epoch=train_epoch, iter_num=None, train=False)
            prediction.append(preds)
            real_value.append(testy)
        prediction = torch.cat(prediction,dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # summarize the results.
        ## test performance of different horizon
        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = prediction[:,i,:,:]
            real    = real_value[:,i,:,:]
            # metrics
            metric_results = {}
            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(pred, real, null_val=self.null_val)
                metric_results[metric_name] = metric_item.item()
            log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            log     = log.format(i+1, metric_results['MAE'], metric_results['RMSE'], metric_results['MAPE'])
            self.logger.info(log)
        ## test performance overall
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, null_val=self.null_val)
            self.update_epoch_meter('test_'+metric_name, metric_item.item())
            metric_results[metric_name] = metric_item.item()

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val_MAE', greater_best=False)
