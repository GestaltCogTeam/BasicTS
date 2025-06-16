from .launcher import launch_evaluation, launch_training
from .runners import BaseEpochRunner

__version__ = '0.5.1'

__all__ = ['__version__', 'launch_training', 'launch_evaluation', 'BaseEpochRunner']
