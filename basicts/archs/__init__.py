import os

from .builder import build_arch, ARCH_REGISTRY
from ..utils.registry import scan_modules

__all__ = ['build_arch', 'ARCH_REGISTRY']

scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])
