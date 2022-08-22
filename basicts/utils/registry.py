# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
# Modified from: https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/registry.py
# pyre-ignore-all-errors[2,3]
import os
import importlib
from typing import Any, Dict, Iterable, Iterator, Tuple, List

from .misc import scan_dir


__all__ = ['Registry', 'scan_modules']


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None, name: str = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                _name = func_or_class.__name__ if name is None else name
                self._do_register(_name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        _name = obj.__name__ if name is None else name
        self._do_register(_name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        return "Registry of {}:\n".format(self._name) + str(self._obj_map)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


def scan_modules(work_dir: str, file_dir: str, exclude_files: List[str] = []):
    """
    automatically scan and import modules for registry
    """
    module_dir = os.path.dirname(os.path.abspath(file_dir))
    import_prefix = module_dir[module_dir.find(work_dir) + len(work_dir) + 1:].replace('/', '.').replace('\\', '.')

    model_file_names = [
        v[:v.find('.py')].replace('/', '.').replace('\\', '.') \
        for v in scan_dir(module_dir, suffix='py', recursive=True) if v not in exclude_files
    ]

    # import all modules
    return [importlib.import_module(f'{import_prefix}.{file_name}') for file_name in model_file_names]
