from .launcher import launch_evaluation, launch_inference, launch_training
from .runners import BaseEpochRunner

__version__ = '0.4.6.4'

__all__ = ['__version__', 'launch_training', 'launch_evaluation', 'BaseEpochRunner', 'launch_inference']
