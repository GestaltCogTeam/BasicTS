import torch.nn.functional as F

from ..utils import check_nan_inf


def l1_loss(input_data, target_data):
    """unmasked mae."""

    return F.l1_loss(input_data, target_data)


def l2_loss(input_data, target_data):
    """unmasked mse"""

    check_nan_inf(input_data)
    check_nan_inf(target_data)
    return F.mse_loss(input_data, target_data)
