import torch
from torch.nn import HuberLoss


def masked_huber(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor, reduction='mean', delta=1.0) -> torch.Tensor:
    """
    Calculate the Masked Huber Loss between predicted and target values,
    ignoring entries in the target tensor that match the specified null value.

    The Huber Loss is a combination of the mean squared error and the mean absolute error,
    making it less sensitive to outliers in the data.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        delta (float, optional): Specifies the threshold at which to change between delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked Huber loss.
    """

    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)

    mask = mask.float()
    prediction, targets = prediction * mask, targets * mask

    prediction = torch.nan_to_num(prediction)
    targets = torch.nan_to_num(targets)

    loss = HuberLoss(reduction, delta)(prediction, targets)

    return loss
