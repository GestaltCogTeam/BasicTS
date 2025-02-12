import torch
import numpy as np
from basicts.metrics import masked_mse
import pdb 


def station_loss(y, statistics_pred, period_len):
    bs, length, dim = y.shape
    y = y.reshape(bs, -1, period_len, dim)
    mean = torch.mean(y, dim=2)
    std = torch.std(y, dim=2)
    station_ture = torch.cat([mean, std], dim=-1)
    loss = masked_mse(statistics_pred, station_ture)
    return loss

def san_loss(prediction, target, statistics_pred, period_len, epoch, station_pretrain_epoch, train):
    if train:
        if epoch + 1 <= station_pretrain_epoch:
            return station_loss(target.squeeze(-1), statistics_pred, period_len)

        else:
            return masked_mse(prediction, target)
    else:
        return masked_mse(prediction, target)

