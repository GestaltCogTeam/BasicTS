# define more learning rate shedulers here

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ['CosineWarmup', 'CosineWarmupRestarts']


class CosineWarmup(LambdaLR):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    
    Modified from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/optimization.py#L144

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def __init__(self,  optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        lr_lambda = partial(
                self._get_cosine_schedule_with_warmup_lr_lambda,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class CosineWarmupRestarts(LambdaLR):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    # Modified from https://github.com/huggingface/transformers/blob/c2820c94916e34baf4486accae74760972183a2f/src/transformers/optimization.py#L144

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1):
        lr_lambda = partial(
                self._get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

class SANWarmupMultiStepLR(LambdaLR):
    """
    A learning rate scheduler that uses a fixed learning rate during a warmup phase,
    then switches to the original learning rate and follows a MultiStepLR schedule.

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        warmup_epochs (`int`):
            The number of epochs for the warmup phase.
        warmup_lr (`float`):
            The fixed learning rate during the warmup phase.
        milestones (`list[int]`):
            List of epoch indices for MultiStepLR. Must be increasing.
        gamma (`float`, optional, default=0.1):
            Multiplicative factor of learning rate decay.
        last_epoch (`int`, optional, default=-1):
            The index of the last epoch when resuming training.
    """
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, warmup_lr: float, milestones: list, gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = milestones
        self.gamma = gamma
        base_lr = optimizer.defaults['lr']  # 原始学习率
        # lr_lambda = lambda epoch: self._get_lr_lambda(epoch, warmup_epochs, warmup_lr, base_lr, milestones, gamma)
        lr_lambda = partial(
                self._get_lr_lambda,
                warmup_epochs=warmup_epochs,
                warmup_lr=warmup_lr,
                base_lr=base_lr,
                milestones=milestones,
                gamma=gamma
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_lr_lambda(epoch: int, warmup_epochs: int, warmup_lr: float, base_lr: float, milestones: list, gamma: float):
        if epoch +1 <= warmup_epochs:
            # Warmup phase
            return warmup_lr / base_lr # From SAN: * (0.5 ** ((epoch - 1) // 1))
        # MultiStepLR phase: decay learning rate at milestones
        adjusted_lr = 1.0
        for milestone in milestones:
            if epoch >= milestone:
                adjusted_lr *= gamma
        return adjusted_lr
