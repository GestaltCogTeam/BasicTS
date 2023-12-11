import numpy as np
import torch

def mase(prediction: torch.Tensor, inputs:torch.Tensor, target: torch.Tensor, frequency: int = 1, null_val: float = np.nan) -> np.ndarray:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()
    a = np.mean(np.abs(prediction - target))
    b = np.mean(np.abs(inputs[:-frequency] - inputs[frequency:]))
    result = torch.Tensor([a / b])
    return result
