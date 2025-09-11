import numpy as np
from torch import nn
from basicts.metrics import masked_mae

def step_loss(prediction, target, pred_adj, prior_adj, gsl_coefficient, null_val=np.nan):
    # graph structure learning loss
    B, N, N = pred_adj.shape
    pred_adj = pred_adj.view(B, N*N)
    tru = prior_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(pred_adj, tru)
    # prediction loss
    loss_pred = masked_mae(prediction=prediction, target=target, null_val=null_val)
    # final loss
    loss = loss_pred + loss_graph * gsl_coefficient
    return loss
