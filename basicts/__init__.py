from .launcher import launch_training, launch_evaluation
from .runners import BaseEpochRunner

__version__ = '0.4.4.2'

__all__ = ['__version__', 'launch_training', 'launch_evaluation', 'BaseEpochRunner']
