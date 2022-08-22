import os

from .registry import ARCH_REGISTRY
from easytorch.utils.registry import scan_modules

__all__ = ['ARCH_REGISTRY']

scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])
