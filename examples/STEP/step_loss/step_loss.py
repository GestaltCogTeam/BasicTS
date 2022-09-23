import torch
import numpy as np
from torch import nn
from basicts.losses import masked_mae

def step_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, null_val=np.nan):
    # graph structure learning loss
    B, N, N = theta.shape
    theta = theta.view(B, N*N)
    tru = priori_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(theta, tru)
    # prediction loss
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    # final loss
    loss = loss_pred + loss_graph * gsl_coefficient
    return loss
