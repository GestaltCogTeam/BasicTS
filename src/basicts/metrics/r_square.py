import torch


def masked_r2(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the Masked R square between the predicted and target values,
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
    prediction, targets = prediction * mask, targets * mask

    prediction = torch.nan_to_num(prediction)
    targets = torch.nan_to_num(targets)

    ss_res = torch.sum(torch.pow((targets - prediction), 2), dim=1)
    ss_tot = torch.sum(torch.pow(targets - torch.mean(targets, dim=1, keepdim=True), 2), dim=1)

    # 计算 R^2
    loss = 1 - (ss_res / (ss_tot + 1e-6))

    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero
    return torch.mean(loss)
