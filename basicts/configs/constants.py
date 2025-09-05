from enum import Enum


class BASICTS_TASK(Enum):
    SPATIAL_TEMPORAL_FORECASTING = 'spatial_temporal_forecasting'
    TIME_SERIES_FORECASTING = 'time_series_forecasting'
    TIME_SERIES_IMPUTATION = 'time_series_imputation'
    TIME_SERIES_CLASSIFICATION = 'time_series_classification'
    TIME_SERIES_ANOMALY_DETECTION  = 'time_series_anomaly_detection'
