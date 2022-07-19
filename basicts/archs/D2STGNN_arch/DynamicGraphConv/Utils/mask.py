import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask   = model_args['adjs']
    
    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
