import torch


def accuracy(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate the accuracy of predictions.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        targets (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.

    Returns:
        torch.Tensor: A scalar tensor representing the accuracy.
    """
    return (prediction == targets).float().mean()

