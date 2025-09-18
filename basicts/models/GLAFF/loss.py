import numpy as np

from basicts.metrics import masked_mse


def glaff_loss(plugin_prediction, plugin_target, null_val=np.nan):
    """GLAFF loss function.
        NOTE: During the validation/test/inference phase, the return value of this loss function is meaningless because the target is None.

    Args:
        plugin_prediction (torch.Tensor): prediction from plugin with shape [B, L, N, C]
        plugin_target (torch.Tensor): target from plugin with shape [B, L, N, C]
        null_val (int, optional): null value in the data. Defaults to 0.

    Returns:
        torch.Tensor: loss
    """

    loss = masked_mse(plugin_prediction, plugin_target, null_val)
    return loss
