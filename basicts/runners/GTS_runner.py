from typing import Tuple, Union
import torch
from basicts.runners.short_mts_runner import MTSRunner
from basicts.utils.registry import SCALER_REGISTRY
from easytorch.utils.dist import master_only

"""
    TODO: 
    模块化train_iters, val_iters, and test_iters中的过程。
    否则就会像GTS一样, 一旦模型有一点特殊 (例如多一个返回和不同的loss), 就必须重写整个train_iters, val_iters, and test_iters。
"""

class GTSRunner(MTSRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def setup_graph(self, data):
        try:
            self.train_iters(data, 0, 0)
        except:
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
        prediction_data, pred_adj, prior_adj = self.model(history_data=history_data, future_data=future_data_, batch_seen=iter_num, epoch=epoch)   # B, L, N, C
        assert list(prediction_data.shape)[:3] == [B, L, N], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.data_i_reshape(prediction_data)
        real_value = self.data_i_reshape(future_data)
        return prediction, real_value, pred_adj, prior_adj

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
        prediction, real_value, pred_adj, prior_adj = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            loss = self.loss(prediction[:, :cl_length, :, :], real_value[:, :cl_length, :, :], null_val=self.null_val)
        else:
            loss = self.loss(prediction, real_value, null_val=self.null_val)
        # graph structure loss
        prior_label = prior_adj.view(prior_adj.shape[0] * prior_adj.shape[1]).to(pred_adj.device)
        pred_label  = pred_adj.view(pred_adj.shape[0] * pred_adj.shape[1])
        graph_loss_function  = torch.nn.BCELoss()
        loss_g      = graph_loss_function(pred_label, prior_label)
        loss = loss + loss_g
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
        prediction, real_value, pred_adj, prior_adj = self.forward(data=data, epoch=None, iter_num=None, train=False)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler['func'])(prediction, **self.scaler['args'])
        real_value = SCALER_REGISTRY.get(self.scaler['func'])(real_value, **self.scaler['args'])
        # loss
        loss  = self.loss(prediction, real_value, null_val=self.null_val)
        # graph structure loss
        prior_label = prior_adj.view(prior_adj.shape[0] * prior_adj.shape[1]).to(pred_adj.device)
        pred_label  = pred_adj.view(pred_adj.shape[0] * pred_adj.shape[1])
        graph_loss_function  = torch.nn.BCELoss()
        loss_g      = graph_loss_function(pred_label, prior_label)
        loss = loss + loss_g

        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = metric_func(prediction, real_value, null_val=self.null_val)
            self.update_epoch_meter('val_'+metric_name, metric_item.item())
        return loss

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: int = None):
        """test model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        # test loop
        prediction = []
        real_value  = []
        for iter_index, data in enumerate(self.test_data_loader):
            preds, testy, pred_adj, prior_adj = self.forward(data=data, epoch=train_epoch, iter_num=None, train=False)
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
