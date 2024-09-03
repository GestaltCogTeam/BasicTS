import time
from functools import partial

import torch


class partial_func(partial):
    """
    Custom partial function class that provides a cleaner string representation.

    This prevents the address of the function from being included, which can cause issues with hashing.
    """

    def __str__(self):
        return f"partial({self.func.__name__}, {self.keywords})"

def clock(func):
    """
    Decorator to measure the execution time of a function.

    This decorator prints the time taken for a function to execute.
    """

    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"{func.__name__}: {elapsed:.8f}s")
        return result
    return clocked

def check_nan_inf(tensor: torch.Tensor, raise_ex: bool = True) -> tuple:
    """
    Check for NaN or Inf values in a tensor.

    Args:
        tensor (torch.Tensor): Input tensor to check.
        raise_ex (bool, optional): Whether to raise an exception if NaN or Inf values are found. Defaults to True.

    Raises:
        ValueError: If raise_ex is True and NaN or Inf values are found.

    Returns:
        tuple: A dictionary indicating presence of NaN and Inf values, and a boolean indicating whether either is present.
    """

    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))

    if raise_ex and (nan or inf):
        raise ValueError({"nan": nan, "inf": inf})

    return {"nan": nan, "inf": inf}, nan or inf

def remove_nan_inf(tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove NaN and Inf values from a tensor by replacing them with zeros.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with NaN and Inf values replaced by zeros.
    """

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor
