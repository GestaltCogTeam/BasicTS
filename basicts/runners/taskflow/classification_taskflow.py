from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSClassificationTaskFlow(BasicTSTaskFlow):
    """Classification Task Flow"""

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # mask
        inputs_mask = null_val_mask(data['inputs'], runner.cfg.null_val)

        if runner.scaler is not None:
            scaled_inputs = runner.scaler.transform(data['inputs'], inputs_mask)
            # transform null values to cfg.null_to_num. default: 0.0.
            data['inputs'] = torch.where(inputs_mask, scaled_inputs,
                                         torch.tensor(runner.cfg.null_to_num, device=scaled_inputs.device))

        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        forward_return['logits'] = forward_return['prediction']
        forward_return['prediction'] = torch.argmax(forward_return['logits'], dim=-1)
        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """Get the weight of the forward return"""
        return forward_return['targets'].shape[0]
