import torch

from .mse import masked_mse


def masked_rmse(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the Masked Root Mean Squared Error (RMSE) between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    This function is useful for evaluating model performance on datasets where some target values
    may be missing or irrelevant (denoted by `null_val`). The RMSE provides a measure of the average
    magnitude of errors, accounting only for the valid, non-null entries.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        targets (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        targets_mask (torch.Tensor, optional): The mask tensor with the same shape as `targets`. 
            Defaults to `None`. The function will ignore all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked root mean squared error.
    """

    return torch.sqrt(masked_mse(prediction=prediction, targets=targets, targets_mask=targets_mask))
