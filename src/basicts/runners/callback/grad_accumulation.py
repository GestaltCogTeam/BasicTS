from typing import TYPE_CHECKING

import torch

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners import BasicTSRunner


class GradAccumulation(BasicTSCallback):

    """
    Gradient accumulation callback.
    """

    def __init__(self, accumulation_steps: int):

        self.accumulation_steps = accumulation_steps
        self.current_steps = 0

    def on_train_start(self, runner: "BasicTSRunner"):
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        batch_size = runner.cfg.train_batch_size
        effective_batch_size = batch_size * world_size * self.accumulation_steps
        runner.logger.info(f"Use gradient accumulation with step {self.accumulation_steps}.")
        runner.logger.info(f"Effective batch size: {effective_batch_size}.")

    def on_backward(self, runner: "BasicTSRunner", loss: torch.Tensor):
        runner.should_optimizer_step = (self.current_steps + 1) % self.accumulation_steps == 0
        loss = loss / self.accumulation_steps
        self.current_steps += 1
