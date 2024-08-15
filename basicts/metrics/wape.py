import torch
import numpy as np


def masked_wape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked weighted absolute percentage error (WAPE)

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    prediction, target = prediction * mask, target * mask
    
    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)

    loss =  torch.sum(torch.abs(prediction-target)) / (torch.sum(torch.abs(target)) + 5e-5)
    return torch.mean(loss)
