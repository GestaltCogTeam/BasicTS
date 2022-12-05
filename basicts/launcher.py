from typing import Dict, Union

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
    try:
        easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
    except TypeError:
        # fit higher easy-torch version
        easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
