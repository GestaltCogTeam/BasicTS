import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the accuracy of predictions.

    Args:
        pred (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `pred`.

    Returns:
        torch.Tensor: A scalar tensor representing the accuracy.
    """
    return (pred == target).float().mean()

