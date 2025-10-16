from typing import TYPE_CHECKING

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners import BasicTSRunner


class NoBP(BasicTSCallback):

    """
    Disable backpropagation.
    """

    def on_train_start(self, runner: "BasicTSRunner"):
        runner.should_backward = False
