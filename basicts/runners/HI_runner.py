import time
import math
import torch
from torch import nn
import numpy as np
from basicts.runners.base_runner import BaseRunner
from basicts.utils.registry import SCALER_REGISTRY
from basicts.utils.serialization import load_pkl
from easytorch.easytorch.utils.dist import master_only

class HIRunner(BaseRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.dataset_name = cfg['DATASET_NAME']
        self.null_val = cfg['TRAIN'].get('NULL_VAL', np.nan)        # different datasets have different null_values. For example, 0.0 in traffic speed dataset, nan in traffic flow dataset.
        self.dataset_type = cfg['DATASET_TYPE']
        # read scaler for re-normalization
        self.scaler = load_pkl("datasets/" + self.dataset_name + "/scaler.pkl")
        # define loss
        self.loss = cfg['TRAIN']['LOSS']
        # define metric
        self.metrics = cfg['METRICS']

    def init_training(self, cfg):
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
        
        self.itera_per_epoch = math.ceil(len(dataset) / batch_size)
        
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

    def data_reshaper(self, data: torch.Tensor) -> torch.Tensor:
        """reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data.

        Returns:
            torch.Tensor: reshaped data
        """
        return data

    def train_iters(self, epoch, iter_index, data):
        """Training details.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """
        iter_num = (epoch-1) * self.itera_per_epoch + iter_index

        # preprocess
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)
        future_data     = self.to_running_device(future_data)
        history_data    = self.data_reshaper(history_data)
        future_data     = self.data_reshaper(future_data)

        # feed forward
        prediction_data = self.model(history_data=history_data, batch_seen=iter_num, epoch=epoch)
        # post process
        prediction_data = prediction_data[..., 0]
        future_data     = future_data[..., 0]
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction_data, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(future_data, **self.scaler['args'])
        # loss
        loss = self.loss(prediction, real_value, self.null_val)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, self.null_val)
            self.update_epoch_meter('train_'+metric_name, metric_item.item())

        return loss

    def val_iters(self, iter_index, data):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """
        # preprocess
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)
        future_data     = self.to_running_device(future_data)
        history_data    = self.data_reshaper(history_data)
        future_data     = self.data_reshaper(future_data)

        # feed forward
        prediction_data = self.model(history_data=history_data)
        # post process
        prediction_data = prediction_data[..., 0]
        future_data     = future_data[..., 0]
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction_data, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(future_data, **self.scaler['args'])
        # loss
        mae  = self.loss(prediction, real_value, self.null_val)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, self.null_val)
            self.update_epoch_meter('val_'+metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self, cfg: dict = None, train_epoch: int = None):
        """test model.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test loop
        prediction = []
        real_value  = []
        for iter_index, data in enumerate(self.test_data_loader):
            preds, testy = self.test_iters(iter_index, data)
            prediction.append(preds)
            real_value.append(testy)
        prediction = torch.cat(prediction,dim=0)
        real_value = torch.cat(real_value, dim=0)
        # post process
        prediction = prediction[..., 0]
        real_value = real_value[..., 0]
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # summarize the results.
        # test performance of different horizon
        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = prediction[:,:,i]
            real    = real_value[:,:,i]
            # metrics
            metric_results = {}
            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(pred, real, self.null_val)
                self.update_epoch_meter('test_'+metric_name, metric_item.item())
                metric_results[metric_name] = metric_item.item()
            log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            log     = log.format(i+1, metric_results['MAE'], metric_results['RMSE'], metric_results['MAPE'])
            self.logger.info(log)

        test_end_time = time.time()
        self.update_epoch_meter('test_time', test_end_time - test_start_time)
        # print val meters
        self.print_epoch_meters('test')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        self.on_test_end()

    def test_iters(self, iter_index: int, data: torch.Tensor or tuple):
        future_data, history_data = data
        history_data    = self.to_running_device(history_data)
        future_data     = self.to_running_device(future_data)
        history_data    = self.data_reshaper(history_data)
        future_data     = self.data_reshaper(future_data)

        # feed forward
        prediction_data = self.model(history_data=history_data)
        return prediction_data, future_data

    def backward(self, loss: torch.Tensor):
        """Backward and update params.

        Args:
            loss (torch.Tensor): loss
        """
        pass
        # self.optim.zero_grad()
        # loss.backward()
        # if hasattr(self, 'clip'):
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        # self.optim.step()