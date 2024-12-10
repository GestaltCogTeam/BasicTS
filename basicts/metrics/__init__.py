from .mae import masked_mae
from .mape import masked_mape
from .mse import masked_mse
from .rmse import masked_rmse
from .wape import masked_wape
from .smape import masked_smape
from .r_square import masked_r2
from .corr import masked_corr

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr
            }

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'ALL_METRICS'
]