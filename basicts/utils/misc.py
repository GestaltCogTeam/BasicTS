import time

import torch


def clock(func):
    """clock decorator"""
    def clocked(*args, **kw):
        """decorator for clock"""
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print("%s: %0.8fs..." % (name, elapsed))
        return result
    return clocked


def check_nan_inf(tensor: torch.Tensor, raise_ex: bool = True) -> tuple:
    """check nan and in in tensor

    Args:
        tensor (torch.Tensor): Tensor
        raise_ex (bool, optional): If raise exceptions. Defaults to True.

    Raises:
        Exception: If raise_ex is True and there are nans or infs in tensor, then raise Exception.

    Returns:
        dict: {'nan': bool, 'inf': bool}
        bool: if exist nan or if
    """

    # nan
    nan = torch.any(torch.isnan(tensor))
    # inf
    inf = torch.any(torch.isinf(tensor))
    # raise
    if raise_ex and (nan or inf):
        raise Exception({"nan": nan, "inf": inf})
    return {"nan": nan, "inf": inf}, nan or inf


def remove_nan_inf(tensor: torch.Tensor):
    """remove nan and inf in tensor

    Args:
        tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor
