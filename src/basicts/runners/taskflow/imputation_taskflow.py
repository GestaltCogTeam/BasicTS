from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask, reconstruction_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSImputationTaskFlow(BasicTSTaskFlow):
    """Imputation Task Flow"""

    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # mask null value
        inputs_null_mask = null_val_mask(data['inputs'], runner.cfg.null_val)

        # normalize inputs
        if runner.scaler is not None:
            data['inputs'] = runner.scaler.transform(data['inputs'], inputs_null_mask)

        # transform null values to cfg.null_to_num. default: 0.0.
        data['inputs'] = torch.where(inputs_null_mask, data['inputs'],
                                    torch.tensor(runner.cfg.null_to_num, device=data['inputs'].device))

        # ground truth for self-supervised imputation
        data['targets'] = data['inputs']
        # mask for self-supervised reconstruction
        inputs_rec_mask = reconstruction_mask(data['inputs'], runner.cfg.mask_ratio)
        data['inputs'] = data['inputs'] * inputs_rec_mask
        data['targets_mask'] = inputs_null_mask * ~inputs_rec_mask

        return data

    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        # inverse transform
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return['prediction'] = runner.scaler.inverse_transform(forward_return['prediction'])
        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """Get the weight of the forward return"""

        return (forward_return['targets_mask']).sum().item()
