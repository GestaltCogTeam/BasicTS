import functools
import random
from typing import Any, Callable, Dict, Optional, Union

import torch


def distributed(
    default_node_num: int = 1,
    default_device_num: int = 0,
    default_node_rank: int = 0,
    default_dist_backend: Optional[Union[str, torch.distributed.Backend]] = None,
    default_init_method: Optional[str] = None,
):
    """Decorator to convert a training function into a distributed function.

    Usage:
        >>> class Runner:
        >>>     @distributed(default_node_num=2, default_device_num=8)
        >>>     def train(self, a, b, **kwargs):
        >>>         print(f"Training with: {a}, {b}")
        >>>
        >>> runner = Runner()
        >>> runner.train(1, 2, node_num=4)  # Override default_node_num
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract distributed params from kwargs (if provided by user)
            node_num = kwargs.pop('node_num', default_node_num)
            device_num = kwargs.pop('device_num', default_device_num)
            node_rank = kwargs.pop('node_rank', default_node_rank)
            dist_backend = kwargs.pop('dist_backend', default_dist_backend)
            init_method = kwargs.pop('init_method', default_init_method)

            # Validate params
            if node_num < 1:
                raise ValueError('node_num must be >= 1')
            if device_num < 1:
                raise ValueError('device_num must be >= 1')
            if node_rank >= node_num:
                raise ValueError('node_rank must be < node_num')

            word_size = node_num * device_num

            if word_size == 1:
                # Single GPU, no distribution
                return func(self, *args, **kwargs)
            else:
                # Distributed training
                dist_backend = dist_backend or 'nccl'
                if init_method is None:
                    if node_num == 1:
                        init_method = f'tcp://127.0.0.1:{random.randint(50000, 65000)}'
                    else:
                        raise ValueError('init_method must be provided for multi-node')

                dist_params = {
                    'device_type': 'cuda',
                    'device_num': device_num,
                    'node_rank': node_rank,
                    'word_size': word_size,
                    'dist_backend': dist_backend,
                    'init_method': init_method,
                }

                torch.multiprocessing.spawn(
                    _distributed_worker,
                    args=(dist_params, func, (self,) + args, kwargs),
                    nprocs=device_num,
                    join=True,
                )

        return wrapper
    return decorator

def _distributed_worker(local_rank: int, dist_params: Dict[str, Any], func: Callable, *args):
    """Distributed function for `torch.multiprocessing.spawn`

    Args:
        local_rank (int): Local rank of current process group.
        dist_params (Dict[str, Any]): Other distributed parameters.
        func (Callable): A function.
    """

    logger = get_logger('easytorch-launcher')

    rank = dist_params['device_num'] * dist_params['node_rank'] + local_rank
    logger.info(
        'Launching in distributed mode. Distributed parameters:'\
        'word_size={:d}, node_rank={:d}, rank={:d}, local_rank={:d}, dist_backend={}, init_method={}'.format(
            dist_params['word_size'], dist_params['node_rank'], rank, local_rank,
            dist_params['dist_backend'], dist_params['init_method']
        )
    )

    set_device_type(dist_params['device_type'])

    torch.distributed.init_process_group(
        backend=dist_params['dist_backend'],
        init_method=dist_params['init_method'],
        rank=rank,
        world_size=dist_params['word_size']
    )

    set_device(local_rank)

    args, kwargs = args
    func(*args, **kwargs)
