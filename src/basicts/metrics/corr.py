import torch


def masked_corr(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Masked Pearson Correlation Coefficient between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """

    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    prediction_mean = torch.mean(prediction, dim=1, keepdim=True)
    target_mean = torch.mean(targets, dim=1, keepdim=True)

    # Compute the deviation of prediction and target from their means
    prediction_dev = prediction - prediction_mean
    target_dev = targets - target_mean

    # Compute the Pearson Correlation Coefficient
    numerator = torch.sum(prediction_dev * target_dev, dim=1, keepdim=True)
    denominator = torch.sqrt(torch.sum(prediction_dev ** 2, dim=1, keepdim=True) * torch.sum(target_dev ** 2, dim=1, keepdim=True))  # 分母
    loss = numerator / denominator

    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)
