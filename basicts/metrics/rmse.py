import numpy as np
import torch

from .mse import masked_mse


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Root Mean Squared Error (RMSE) between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    This function is useful for evaluating model performance on datasets where some target values
    may be missing or irrelevant (denoted by `null_val`). The RMSE provides a measure of the average
    magnitude of errors, accounting only for the valid, non-null entries.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will ignore all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked root mean squared error.
    """

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))
