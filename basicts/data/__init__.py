import os
import platform

from easytorch.utils.registry import scan_modules

from .registry import SCALER_REGISTRY
from .dataset import TimeSeriesForecastingDataset

__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset"]

project_dir = os.getcwd()
if platform.system().lower() == 'windows':
    # On Windows systems, os.getcwd() will get an uppercase drive letter, such as C:\\Users\\...
    # However, the drive letter obtained by __file__ is lowercase, such as c:\\Users\\...
    # TODO: remove these code when this issue is officially fixed in the next EasyTorch version.
    project_dir = project_dir[0].lower() + project_dir[1:]

scan_modules(project_dir, __file__, ["__init__.py", "registry.py"])
