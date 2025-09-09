from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.runners.taskflow import BasicTSTaskFlow
from basicts.utils import null_val_mask

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSForecastingTaskFlow(BasicTSTaskFlow):
    """Forecasting Task Flow"""

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # mask
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['target'], runner.cfg.null_val)

        if runner.scaler is not None:
            scaled_inputs = runner.scaler.transform(data['inputs'], inputs_mask)
            scaled_targets = runner.scaler.transform(data['target'], targets_mask)

            # transform null values to cfg.null_to_num. default: 0.0.
            scaled_inputs = torch.where(inputs_mask, scaled_inputs,
                                        torch.tensor(runner.cfg.null_to_num, device=scaled_inputs.device))
            scaled_targets = torch.where(targets_mask, scaled_targets,
                                        torch.tensor(runner.cfg.null_to_num, device=scaled_targets.device))
            data['inputs'] = scaled_inputs
            data['target'] = scaled_targets

        data['mask'] = targets_mask # added by preprocessing, will be used in loss function
        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # inverse transform
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(forward_return['prediction'])

        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """Get the weight of the forward return"""

        return forward_return['mask'].sum().item()
