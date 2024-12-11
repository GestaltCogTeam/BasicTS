import numpy as np
import torch


def masked_corr(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
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

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    prediction_mean = torch.mean(prediction, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=1, keepdim=True)

    # 计算偏差 (X - mean_X) 和 (Y - mean_Y)
    prediction_dev = prediction - prediction_mean
    target_dev = target - target_mean

    # 计算皮尔逊相关系数
    numerator = torch.sum(prediction_dev * target_dev, dim=1, keepdim=True)  # 分子
    denominator = torch.sqrt(torch.sum(prediction_dev ** 2, dim=1, keepdim=True) * torch.sum(target_dev ** 2, dim=1, keepdim=True))  # 分母
    loss = numerator / denominator

    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)
