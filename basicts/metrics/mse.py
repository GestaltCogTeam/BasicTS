import numpy as np
import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Squared Error (MSE) between predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is useful for scenarios where the dataset contains missing or irrelevant values 
    (denoted by `null_val`) that should not contribute to the loss calculation. The function applies 
    a mask to these values, ensuring they do not affect the error metric.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean squared error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)  # Return the mean of the masked loss
