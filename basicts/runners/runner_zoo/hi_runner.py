import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class HIRunner(SimpleTimeSeriesForecastingRunner):

    def backward(self, loss: torch.Tensor):
        pass
        return
