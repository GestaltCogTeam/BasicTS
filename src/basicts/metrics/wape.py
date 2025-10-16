import torch


def masked_wape(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
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

    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)
    mask = mask.float()
    prediction, targets = prediction * mask, targets * mask

    prediction = torch.nan_to_num(prediction)
    targets = torch.nan_to_num(targets)

    loss = torch.sum(torch.abs(prediction - targets), dim=1) / (torch.sum(torch.abs(targets), dim=1) + 5e-5)

    return torch.mean(loss)
