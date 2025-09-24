from typing import TYPE_CHECKING, Optional

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class EarlyStopping(BasicTSCallback):

    """
    Early stopping callback.

    Args:
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter: int = 0

    def on_train_start(self, runner: 'BasicTSRunner'):
        runner.logger.info(f'Use early stopping with patience {self.patience}.')

    def on_validate_end(self, runner: 'BasicTSRunner', train_step: int, train_epoch: Optional[int] = None):
        metric = runner.meter_pool.get_value(f'val/{runner.target_metric}')
        best_metric = runner.best_metrics.get(f'val/{runner.target_metric}')
        if best_metric is not None and (metric >= best_metric if runner.metrics_best == 'min' else metric <= best_metric):
            self.counter += 1
            if self.counter >= self.patience:
                if runner.training_unit == 'epoch':
                    runner.logger.info(f'Early stopping at epoch {train_epoch}.')
                elif runner.training_unit == 'step':
                    runner.logger.info(f'Early stopping at step {train_step}.')
                runner.should_training_stop = True
        else:
            self.counter = 0
