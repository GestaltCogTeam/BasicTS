import torch
import numpy as np
from basicts.losses import masked_mae


def gts_loss(prediction, real_value, pred_adj, prior_adj, null_val = np.nan):
    # graph loss
    prior_label = prior_adj.view(prior_adj.shape[0] * prior_adj.shape[1]).to(pred_adj.device)
    pred_label  = pred_adj.view(pred_adj.shape[0] * pred_adj.shape[1])
    graph_loss_function  = torch.nn.BCELoss()
    loss_g      = graph_loss_function(pred_label, prior_label)
    # regression loss
    loss_r = masked_mae(prediction, real_value, null_val=null_val)
    # total loss
    loss = loss_r + loss_g
    return loss
