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

def precision(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the precision of predictions.

    Args:
        pred (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `pred`.

    Returns:
        torch.Tensor: A scalar tensor representing the precision.
    """
    true_positives = (pred == target).float().sum()
    false_positives = (pred != target).float().sum()
    return true_positives / (true_positives + false_positives)

def recall(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the recall of predictions.

    Args:
        pred (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `pred`.

    Returns:
        torch.Tensor: A scalar tensor representing the recall.
    """
    true_positives = (pred == target).float().sum()
    false_negatives = (pred != target).float().sum()
    return true_positives / (true_positives + false_negatives)

def f1_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the F1 score of predictions.

    Args:
        pred (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `pred`.

    Returns:
        torch.Tensor: A scalar tensor representing the F1 score.
    """
    precision_item = precision(pred, target)
    recall_item = recall(pred, target)
    return 2 * (precision_item * recall_item) / (precision_item + recall_item)
