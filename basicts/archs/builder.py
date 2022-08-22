from copy import deepcopy

from ..utils.registry import Registry

ARCH_REGISTRY = Registry('Arch')


def build_arch(arch_name, params):
    params = deepcopy(params)
    arch = ARCH_REGISTRY.get(arch_name)(**params)
    return arch
