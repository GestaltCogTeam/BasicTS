from easytorch.utils.registry import Registry

SCALER_REGISTRY = Registry('Scaler')

"""
data normalization and re-normalization
"""

# ====================================== re-normalizations ====================================== #
@SCALER_REGISTRY.register()
def re_max_min_normalization(x, **kwargs):
    _min, _max = kwargs['min'], kwargs['max']
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

@SCALER_REGISTRY.register()
def standard_re_transform(x, **kwargs):
    mean, std = kwargs['mean'], kwargs['std']
    x = x * std
    x = x + mean
    return x

# ====================================== normalizations ====================================== #
# omitted to avoid redundancy, as they should only be used in data preprocessing in `scripts/data_preparation`
