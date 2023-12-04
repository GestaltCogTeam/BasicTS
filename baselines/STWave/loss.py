import torch
import numpy as np

from basicts.losses import masked_mae


def stwave_masked_mae(preds: list, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    lloss = masked_mae(preds[...,1:2], preds[...,2:])
    loss = masked_mae(preds[...,:1], labels)

    return loss + lloss
