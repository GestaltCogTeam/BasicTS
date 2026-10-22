from dataclasses import dataclass, field
from typing import List, Optional

from basicts.runners.taskflow import BasicTSAnomalyDetectionTaskFlow, BasicTSTaskFlow

from .tsi_config import BasicTSImputationConfig


@dataclass(init=False)
class BasicTSAnomalyDetectionConfig(BasicTSImputationConfig):
    """Reconstruction-based anomaly detection using continuous BasicTS datasets.

    Select checkpoints by validation reconstruction MSE, then calibrate a score
    quantile on a normal validation split. Test labels are used only for metrics.
    """

    taskflow: BasicTSTaskFlow = field(default_factory=BasicTSAnomalyDetectionTaskFlow)
    mask_ratio: float = 0.0
    loss: str = "MSE"
    metrics: List[str] = field(default_factory=lambda: ["MSE"])
    target_metric: str = "MSE"
    anomaly_ratio: float = 0.01
    anomaly_labels_path: Optional[str] = None

    def __post_init__(self):
        if self.mask_ratio != 0:
            raise ValueError(
                "Anomaly detection uses unmasked reconstruction: mask_ratio must be 0."
            )
        if not 0 < self.anomaly_ratio < 1:
            raise ValueError("anomaly_ratio must be between 0 and 1.")
        super().__post_init__()
