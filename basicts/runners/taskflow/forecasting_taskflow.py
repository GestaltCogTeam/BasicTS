from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSForecastingTaskFlow(BasicTSTaskFlow):
    """Forecasting Task Flow"""

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # mask
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)
        targets_mask = null_val_mask(data['targets'], runner.cfg.null_val)

        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'], inputs_mask)
            data['targets'] = runner.scaler.transform(data['targets'], targets_mask)

        # transform null values to cfg.null_to_num. default: 0.0.
        data['inputs'] = torch.where(inputs_mask, data['inputs'],
                                    torch.tensor(runner.cfg.null_to_num, device=data['inputs'].device))
        data['targets'] = torch.where(targets_mask, data['targets'],
                                    torch.tensor(runner.cfg.null_to_num, device=data['targets'].device))

        data['targets_mask'] = targets_mask # added by preprocessing, will be used in loss function
        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # inverse transform
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(forward_return['prediction'])
            forward_return['targets'] = runner.scaler.inverse_transform(forward_return['targets'], forward_return['targets_mask'])

        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """Get the weight of the forward return"""

        return forward_return['targets_mask'].sum().item()
