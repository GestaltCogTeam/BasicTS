from typing import Dict, Union
from packaging import version

import easytorch


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
