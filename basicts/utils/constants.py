from enum import Enum


class BasicTSEnum(str, Enum):
    """BasicTS Enum"""

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class BasicTSMode(BasicTSEnum):
    """Mode Enum"""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EVAL = "eval"
    INFERENCE = "inference"


class RunnerStatus(BasicTSEnum):
    """Status Enum"""

    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    EVALUATING = "evaluating"
    FINISHED = "finished"


class BasicTSTask(BasicTSEnum):
    """Task Enum"""

    SPATIAL_TEMPORAL_FORECASTING = "spatial_temporal_forecasting"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TIME_SERIES_IMPUTATION = "time_series_imputation"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_ANOMALY_DETECTION = "time_series_anomaly_detection"

