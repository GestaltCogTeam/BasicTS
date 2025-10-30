from typing import TYPE_CHECKING, Iterable

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class AddAuxiliaryLoss(BasicTSCallback):
    """
    Adding auxiliary loss callback.
    """

    def __init__(self, losses: Iterable[str] = None):
        """
        Args:
            losses: Iterable[str], keys of losses in `forward_return` that will be added. Default is ["aux_loss"].
        """
        super().__init__()
        self.losses = losses or ["aux_loss"]

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs):
        forward_return = kwargs["forward_return"]
        regression_loss = runner._metric_forward(runner.loss, forward_return)
        for loss_name in self.losses:
            if loss_name in forward_return:
                regression_loss += forward_return[loss_name]
