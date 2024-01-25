import os
import time
import importlib
from typing import List
from functools import partial

import torch
from easytorch.utils.misc import scan_dir


def scan_modules(work_dir: str, file_dir: str, exclude_files: List[str] = None, exclude_dirs: List[str] = None):
    """
    overwrite easytorch.utils.scan_modeuls: automatically scan and import modules for registry, and exclude some files and dirs.
    """
    module_dir = os.path.dirname(os.path.abspath(file_dir))
    import_prefix = module_dir[module_dir.find(work_dir) + len(work_dir) + 1:].replace("/", ".").replace("\\", ".")

    if exclude_files is None:
        exclude_files = []
    if exclude_dirs is None:
        exclude_dirs = []

    # get all file names, and remove the files in exclude_files
    model_file_names = [
        v[:v.find(".py")].replace("/", ".").replace("\\", ".") \
        for v in scan_dir(module_dir, suffix="py", recursive=True) if v not in exclude_files
    ]

    # remove the files in exclude_dirs. TODO: use os.path to check
    for exclude_dir in exclude_dirs:
        exclude_dir = exclude_dir.replace("/", ".").replace("\\", ".")
        model_file_names = [file_name for file_name in model_file_names if exclude_dir not in file_name]

    # import all modules
    return [importlib.import_module(f"{import_prefix}.{file_name}") for file_name in model_file_names]


class partial_func(partial):
    """partial class.
        __str__ in functools.partial contains the address of the function, which changes randomly and will disrupt easytorch's md5 calculation.
    """

    def __str__(self):
        return "partial({}, {})".format(self.func.__name__, self.keywords)

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
