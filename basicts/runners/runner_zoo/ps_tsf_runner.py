import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class PatchStructuralTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def train_iters(self, epoch: int, iter_index: int, data) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """

        iter_num = (epoch - 1) * self.step_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
        
        forward_return['parameters'] = list(self.model.parameters())
        loss = self.metric_forward(self.loss, forward_return)
        self.update_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_meter(f'train/{metric_name}', metric_item.item())
        return loss