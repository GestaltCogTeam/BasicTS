import time
import torch

def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('%s: %0.8fs...' % (name, elapsed))
        return result
    return clocked

def check_nan_inf(tensor, raise_ex=True):
    # nan
    nan = torch.any(torch.isnan(tensor))
    # inf
    inf = torch.any(torch.isinf(tensor))
    # raise
    if raise_ex and (nan or inf):
        raise Exception({"nan":nan, "inf":inf})
    return {"nan":nan, "inf":inf}, nan or inf

def remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor
