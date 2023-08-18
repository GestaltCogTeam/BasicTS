from packaging import version
from typing import Callable, Dict, Union, Tuple

import easytorch


def launch_runner(cfg: Union[Dict, str], fn: Callable, args: Tuple = (), device_type: str = "gpu", devices: str = None):
    easytorch_version = easytorch.__version__
    if version.parse(easytorch_version) >= version.parse("1.3"):
        easytorch.launch_runner(cfg=cfg, fn=fn, args=args, device_type=device_type, devices=devices)
    else:
        easytorch.launch_runner(cfg=cfg, fn=fn, args=args, gpus=devices)

def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
    """Extended easytorch launch_training.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    # pre-processing of some possible future features, such as:
    # registering model, runners.
    # config checking
    pass
    # launch training based on easytorch
    easytorch_version = easytorch.__version__
    if version.parse(easytorch_version) >= version.parse("1.3"):
        easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
    else:
        easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
