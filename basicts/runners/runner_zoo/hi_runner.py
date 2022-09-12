import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class HIRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
    
    def backward(self, loss: torch.Tensor):
        pass
        return
