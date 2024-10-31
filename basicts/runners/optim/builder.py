# Modified from easytorch: https://github.com/cnstark/easytorch/blob/96c903eb10deb0f96ca16a7b677ad0051b6f33f6/easytorch/core/optimizer_builder.py


from typing import Dict

from torch import nn, optim
from torch.optim import lr_scheduler

from . import lr_schedulers as basicts_lr_scheduler
from . import optimizers as basicts_optim


def build_optim(optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:
    """Build optimizer from `optim_cfg`
    `optim_cfg` is part of config which defines fields about optimizer

    structure of `optim_cfg` is
    {
        'TYPE': (str or type) optimizer name or type, such as ``Adam``, ``SGD``,
            or custom optimizer type.
        'PARAM': (Dict) optimizer init params except first param `params`
    }

    Note:
        Optimizer is initialized by reflection, please ensure optim_cfg['TYPE'] is in `torch.optim`

    Examples:
        optim_cfg = {
            'TYPE': 'Adam',
            'PARAM': {
                'lr': 1e-3,
                'betas': (0.9, 0.99)
                'eps': 1e-8,
                'weight_decay': 0
            }
        }
        An `Adam` optimizer will be built.

    Args:
        optim_cfg (Dict): optimizer config
        model (nn.Module): model defined by user

    Returns:
        optimizer (optim.Optimizer)
    """

    if isinstance(optim_cfg['TYPE'], type):
        optim_type = optim_cfg['TYPE']
    else:
        if hasattr(optim, optim_cfg['TYPE']):
            optim_type = getattr(optim, optim_cfg['TYPE'])
        else:
            optim_type = getattr(basicts_optim, optim_cfg['TYPE'])
    optim_param = optim_cfg['PARAM'].copy()
    optimizer = optim_type(model.parameters(), **optim_param)
    return optimizer


def build_lr_scheduler(lr_scheduler_cfg: Dict, optimizer: optim.Optimizer) -> lr_scheduler._LRScheduler:
    """Build lr_scheduler from `lr_scheduler_cfg`
    `lr_scheduler_cfg` is part of config which defines fields about lr_scheduler

    structure of `lr_scheduler_cfg` is
    {
        'TYPE': (str or type) lr_scheduler name or type, such as ``MultiStepLR``, ``CosineAnnealingLR``,
            or custom lr_scheduler type
        'PARAM': (Dict) lr_scheduler init params except first param `optimizer`
    }

    Note:
        LRScheduler is initialized by reflection, please ensure
        lr_scheduler_cfg['TYPE'] is in `torch.optim.lr_scheduler` or `easytorch.easyoptim.easy_lr_scheduler`,
        if the `type` is not found in `torch.optim.lr_scheduler`,
        it will continue to be search in `easytorch.easyoptim.easy_lr_scheduler`

    Examples:
        lr_scheduler_cfg = {
            'TYPE': 'MultiStepLR',
            'PARAM': {
                'milestones': [100, 200, 300],
                'gamma': 0.1
            }
        }
        An `MultiStepLR` lr_scheduler will be built.

    Args:
        lr_scheduler_cfg (Dict): lr_scheduler config
        optimizer (nn.Module): optimizer

    Returns:
        LRScheduler
    """

    lr_scheduler_cfg['TYPE'] = lr_scheduler_cfg['TYPE']
    if isinstance(lr_scheduler_cfg['TYPE'], type):
        scheduler_type = lr_scheduler_cfg['TYPE']
    else:
        if hasattr(lr_scheduler, lr_scheduler_cfg['TYPE']):
            scheduler_type = getattr(lr_scheduler, lr_scheduler_cfg['TYPE'])
        else:
            scheduler_type = getattr(basicts_lr_scheduler, lr_scheduler_cfg['TYPE'])
    scheduler_param = lr_scheduler_cfg['PARAM'].copy()
    scheduler_param['optimizer'] = optimizer
    scheduler = scheduler_type(**scheduler_param)
    return scheduler
