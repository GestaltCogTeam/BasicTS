import torch
import numpy as np

from basicts.losses import masked_mae


def stwave_masked_mae(prediction: list, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    lloss = masked_mae(prediction[...,1:2], prediction[...,2:], null_val)
    loss = masked_mae(prediction[...,:1], target, null_val)

    return loss + lloss
