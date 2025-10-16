from typing import TYPE_CHECKING, Union

from torch.nn.utils import clip_grad_norm_

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners import BasicTSRunner


class GradientClipping(BasicTSCallback):

    """
    Clip gradient norm.

    Args:
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total norm of the gradients from
            :attr:`parameters` is ``nan``, ``inf``, or ``-inf``. Default: False
        foreach (bool, optional): whether per-parameter gradient clipping should be used instead of the
            default global gradient clipping. If ``None``, global clipping is used. Default: None
    """

    def __init__(self,
                 max_norm: float,
                 norm_type: float = 2,
                 error_if_nonfinite: bool = False,
                 foreach: Union[bool, None] = None,
                 log: bool = False):

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.foreach = foreach
        self.log = log

    def on_train_start(self, runner: "BasicTSRunner"):
        runner.logger.info(f"Use clip grad, max_norm: {self.max_norm}, norm_type: {self.norm_type}, "\
                           f"error_if_nonfinite: {self.error_if_nonfinite}, foreach: {self.foreach}.")
        if self.log:
            runner.register_meter("train/grad_norm", "train", "{:.4f}")

    def on_optimizer_step(self, runner: "BasicTSRunner"):
        runner.amp_scaler.unscale_(runner.optimizer)
        if self.log:
            grad_norm = sum(
                        param.grad.data.norm(2).item() ** 2 for param in runner.model.parameters() if param.grad is not None
                    ) ** 0.5
            runner.update_meter("train/grad_norm", grad_norm)

        clip_grad_norm_(runner.model.parameters(), self.max_norm, self.norm_type, self.error_if_nonfinite, self.foreach)
