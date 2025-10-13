from typing import TYPE_CHECKING

from basicts.utils import RunnerStatus

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class CurriculumLearning(BasicTSCallback):

    """
    Curriculum learning callback.

    Args:
        prediction_length (int): Total prediction length, i.e., `output_len`.
        warm_up_epochs (int, optional): Number of warm-up epochs. Default: 0.
        cl_epochs (int, optional): Number of epochs for each curriculum learning stage. Default: 1.
        step_size (int, optional): Step size for the curriculum learning. Default: 1.

    """

    def __init__(self, prediction_length: int, warm_up_epochs: int = 0, cl_epochs: int = 1, step_size: int = 1):
        self.prediction_length = prediction_length
        self.warm_up_epochs = warm_up_epochs
        self.cl_epochs = cl_epochs
        self.step_size = step_size

    def on_train_start(self, runner: "BasicTSRunner"):
        runner.logger.info("Use curriculum learning.")

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs):
        forward_return = kwargs["forward_return"]
        if runner.status == RunnerStatus.TRAINING:
            cl_length = self.curriculum_learning(epoch=runner.epoch)
            try:
                forward_return["prediction"] = forward_return["prediction"][:, :cl_length, :]
                forward_return["targets"] = forward_return["targets"][:, :cl_length, :]
                forward_return["targets_mask"] = forward_return["targets_mask"][:, :cl_length, :]
            except KeyError as ke:
                raise KeyError("Curriculum learning requires 'prediction' and 'targets' in forward_return.") from ke
            except IndexError as ie:
                raise IndexError("Curriculum learning should be used for forecasting tasks" \
                                "with data in shape [batch_size, seq_len, num_features].") from ie

    def curriculum_learning(self, epoch: int) -> int:
        """
        Curriculum learning schedule.

        Args:
            epoch (int): Current epoch.

        Returns:
            int: Curriculum learning length.
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still in warm-up phase
            cl_length = self.prediction_length
        else:
            progress = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.step_size
            cl_length = min(progress, self.prediction_length)
        return cl_length
