from .cls_metrics import accuracy, f1_score, precision, recall
from .corr import masked_corr
from .mae import masked_mae
from .mape import masked_mape
from .metric_meter import AvgMeter, RMSEMeter
from .mse import masked_mse
from .r_square import masked_r2
from .rmse import masked_rmse
from .smape import masked_smape
from .wape import masked_wape

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score
            }

METRIC_METER = {
    'RMSE': RMSEMeter,
    'default': AvgMeter
}

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'incremental_masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'ALL_METRICS',
    'METRIC_METER'
]