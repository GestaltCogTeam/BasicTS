import numpy as np
import torch


def null_val_mask(data: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Create a mask for the input data.

    Args:
        data (np.ndarray): Input data.
        mask_ratio (float): Ratio of the data to be masked.

    Returns:
        np.ndarray: Mask for the input data.
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(data)
    else:
        eps = 5e-5
        mask = ~torch.isclose(data, torch.tensor(null_val).to(data.device), atol=eps)
    return mask
