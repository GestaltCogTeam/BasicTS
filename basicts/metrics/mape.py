import numpy as np
import torch

# ============== MAPE ================== #
def masked_mape(preds, labels, null_val=np.nan):
    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels<1e-2, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
