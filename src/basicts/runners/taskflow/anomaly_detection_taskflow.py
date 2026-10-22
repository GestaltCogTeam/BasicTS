from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.utils.mask import null_val_mask

from .basicts_taskflow import BasicTSTaskFlow

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSAnomalyDetectionTaskFlow(BasicTSTaskFlow):
    """Unmasked reconstruction with pointwise errors in the scaler space.

    Anomaly labels never enter the reconstruction model or its training loss.
    Threshold calibration and detection metrics run after checkpoint selection.
    """

    def preprocess(
        self, runner: "BasicTSRunner", data: Dict[str, Any]
    ) -> Dict[str, Any]:
        observed = null_val_mask(data["inputs"], runner.cfg.null_val)
        inputs = data["inputs"]
        if runner.scaler is not None:
            inputs = runner.scaler.transform(inputs, observed)
        inputs = torch.where(
            observed, inputs, inputs.new_tensor(runner.cfg.null_to_num)
        )
        data["inputs"] = inputs
        data["targets"] = inputs
        data["targets_mask"] = observed
        return data

    def postprocess(
        self, runner: "BasicTSRunner", forward_return: Dict[str, Any]
    ) -> Dict[str, Any]:
        prediction, targets = forward_return["prediction"], forward_return["targets"]
        observed = forward_return["targets_mask"].bool()
        if prediction.shape != targets.shape:
            raise ValueError(
                "Anomaly detection requires reconstruction with the input shape."
            )
        errors = torch.where(observed, (prediction - targets).square(), 0.0)
        count = observed.sum(dim=-1)
        forward_return["anomaly_scores"] = torch.where(
            count > 0, errors.sum(dim=-1) / count.clamp_min(1), float("nan")
        )
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return["prediction"] = runner.scaler.inverse_transform(prediction)
            forward_return["targets"] = runner.scaler.inverse_transform(
                targets, observed
            )
        return forward_return

    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        return forward_return["targets_mask"].sum().item()
