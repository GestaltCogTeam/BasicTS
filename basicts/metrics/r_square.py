import numpy as np
import torch


def masked_r2(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
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

    eps = 5e-5
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    prediction, target = prediction * mask, target * mask

    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)

    ss_res = torch.sum(torch.pow((target - prediction), 2), dim=1)  # 残差平方和
    ss_tot = torch.sum(torch.pow(target - torch.mean(target, dim=1, keepdim=True), 2), dim=1)  # 总平方和

    # 计算 R^2
    loss = 1 - (ss_res / (ss_tot + eps))

    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero
    return torch.mean(loss)
