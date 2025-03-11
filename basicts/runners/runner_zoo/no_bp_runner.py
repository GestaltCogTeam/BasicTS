import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class NoBPRunner(SimpleTimeSeriesForecastingRunner):

    def backward(self, loss: torch.Tensor):
        pass
