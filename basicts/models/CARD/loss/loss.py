import torch
import torch.nn.functional as F
import numpy as np
from basicts.metrics import masked_mse
import pdb 


def h_loss(outputs, batch_y, ratio):

    h_level_range = [4,8,16,24,48,96]
    for h_level in h_level_range:
        batch,length,channel = outputs.shape
        # print(outputs.shape)
        h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
        h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
        h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
        h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
        h_ratio = ratio[:h_outputs.shape[-2],:]
        # print(h_outputs.shape,h_ratio.shape)
        h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
        h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)

        h_outputs = h_outputs*h_ratio
        h_batch_y = h_batch_y*h_ratio

        h_ouputs_agg *= h_ratio
        h_batch_y_agg *= h_ratio

    loss_1 = F.l1_loss(h_outputs, h_batch_y)*np.sqrt(h_level) / 2 
    loss_2 = F.l1_loss(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) / 2

    return loss_1 + loss_2

def card_loss(prediction, target, use_h_loss, pred_len):
    outputs, batch_y = prediction.squeeze(-1), target.squeeze(-1)

    ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(pred_len)])
    ratio = torch.tensor(ratio).unsqueeze(-1).to(prediction.device)
    outputs = outputs * ratio
    batch_y = batch_y * ratio
    loss = F.l1_loss(prediction, target)

    if not use_h_loss:
        return loss

    else:
        return loss + h_loss(outputs, batch_y, ratio) * 1e-2
        

