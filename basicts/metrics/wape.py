import numpy as np
import torch


def masked_wape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Weighted Absolute Percentage Error (WAPE) between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    WAPE is a useful metric for measuring the average error relative to the magnitude of the target values,
    making it particularly suitable for comparing errors across datasets or time series with different scales.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked weighted absolute percentage error.
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    prediction, target = prediction * mask, target * mask

    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)

    loss = torch.sum(torch.abs(prediction - target), dim=1) / (torch.sum(torch.abs(target), dim=1) + 5e-5)

    return torch.mean(loss)
