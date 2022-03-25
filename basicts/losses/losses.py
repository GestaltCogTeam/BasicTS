from basicts.metrics.mae import masked_mae as maksed_l1_loss

import torch
import torch.nn.functional as F

def L1Loss(input, target, **kwargs):
    return F.l1_loss(input, target)
