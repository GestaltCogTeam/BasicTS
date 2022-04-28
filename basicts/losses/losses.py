from basicts.metrics.mae import masked_mae as masked_l1_loss
from basicts.utils.misc import check_nan_inf
import torch
import torch.nn.functional as F

def L1Loss(input, target, **kwargs):
    return F.l1_loss(input, target)

def MSELoss(input, target, **kwargs):
    check_nan_inf(input)
    check_nan_inf(target)
    return F.mse_loss(input, target)