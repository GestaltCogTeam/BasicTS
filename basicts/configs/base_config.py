import copy
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from numbers import Number
from types import FunctionType, NoneType
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict

from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import BasicTSTaskFlow


@dataclass
class BasicTSConfig(EasyDict):

    """
    Base class for configuration.

    Args:
        d (dict, optional): Dictionary to initialize the configuration. Defaults to None.
    """

    model: torch.nn.Module
    dataset_name: str
    taskflow: BasicTSTaskFlow
    callbacks: List[BasicTSCallback]

    ############################## General Configuration ##############################

    # General settings
    gpus: Optional[str] # Wether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'.
    gpu_num: int # Post-init. Number of GPUs.
    seed: int # Random seed.

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    dataset_type: type
    dataset_params: dict
    batch_size: Optional[int] # if setted, all dataloaders will be setted to the same batch size.
    null_val: float
    null_to_num: float

    # Scaler settings
    scaler: type # Post-init. Scaler.
    norm_each_channel: bool # Post-init. Whether to normalize data for each channel independently.
    rescale: bool # Whether to rescale data. Default: False

    ############################## Model Configuration ##############################

    # Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`.
    # In distributed computing, if there are unused parameters in the forward process, PyTorch usually raises a RuntimeError.
    # In such cases, this parameter should be set to True.
    ddp_find_unused_parameters: bool

    compile_model: bool

    ############################## Metrics Configuration ##############################

    # Metrics settings
    metrics: List[str]
    target_metric: str
    best_metric: Literal['min', 'max']

    ############################## Training Configuration ##############################

    num_epochs: int
    num_steps: int

    # Loss function
    loss: Callable # Loss function

    # Optimizer
    optimizer: type

    # Learning rate scheduler
    lr_scheduler: type

    # Checkpoint loading and saving settings

    # Directory to save checkpoints. Default: 'checkpoints/{model}/{dataset}_{num_epochs}_{input_len}_{output_len}', which will be loaded lazily.
    ckpt_save_dir: str
    # Checkpoint save strategy. `CFG.TRAIN.CKPT_SAVE_STRATEGY` should be None, an int value, a list or a tuple. Default: None.
    # None: remove last checkpoint file every epoch.
    # Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch.
    # List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY, remove last checkpoint file when last_epoch not in ckpt_save_strategy
    ckpt_save_strategy: Union[int, List[int], Tuple[int]]
    finetune_from: str # Checkpoint path for fine-tuning. Default: None. If not specified, the model will be trained from scratch.
    strict_load: bool # Whether to strictly load the checkpoint. Default: True.

    # Train data loader settings
    train_batch_size: int
    train_data_prefetch: bool # Whether to use dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator. Default: False.
    train_data_shuffle: bool # Whether to shuffle the training data. Default: False
    train_data_collate_fn: Callable # Collate function for the training dataloader. Default: None
    train_data_num_workers: int # Number of workers for the training dataloader. Default: 0
    train_data_pin_memory: bool# Whether to pin memory for the training dataloader. Default: False

    ############################## Validation Configuration ##############################

    val_batch_size: int
    val_interval: int # Conduct test every `val_interval` epochs. Default: 1
    val_data_prefetch: bool
    val_data_shuffle: bool
    val_data_collate_fn: Callable
    val_data_num_workers: int
    val_data_pin_memory: bool

    ########################### Evaluation Configuration ##########################

    save_results: bool # Whether to save evaluation results in a numpy file. Default: False

    ############################## Environment Configuration ##############################

    tf32: bool # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
    deterministic: bool # Whether to set the random seed to get deterministic results. Default: False
    cudnn_enabled: bool # Whether to enable cuDNN. Default: True
    cudnn_benchmark: bool # Whether to enable cuDNN benchmark. Default: True
    cudnn_determinstic: bool # Whether to set cuDNN to deterministic mode. Default: False

    ############################## Training-Independent Keys ##############################

    _TRAINING_INDEPENDENT_KEYS: List[str]

    #######################################################################################

    def __init__(self, d = None, **kwargs):
        self._is_initializing = True
        super().__init__(d, **kwargs)
        self._is_initializing = False

    def __str__(self) -> str:
        serialized = {}
        for k, v in self.items():
            # Property, e.g., _runner (field) -> runner (property)
            if k.startswith('_'):
                if hasattr(self, k[1:]):
                    k = k[1:]
                    v = getattr(self, k)
            serialized[k] = self._serialize(v)
        self.md5 = self._get_md5(serialized)
        return json.dumps(serialized, ensure_ascii=False, indent=4)

    def __getitem__(self, key):
        try:
            # for property and LazyField
            return getattr(self, key)
        except AttributeError:
            # compatible with old version
            if isinstance(key, str) and key not in self:
                key = key.replace('.', '_').lower()
            return super().__getitem__(key)

    def save(self):
        json_str = str(self)
        save_dir = os.path.join(self.ckpt_save_dir, self.md5)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'cfg.json'), 'w', encoding='utf-8') as f:
            f.write(json_str)

    def _serialize(self, obj: object) -> object:
        if isinstance(obj, (str, bool, Number, NoneType)):
            return obj
        # List, tuple
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        # Dict
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        # Class or function
        elif isinstance(obj, (type, FunctionType)):
            return obj.__name__
        # Partial function
        elif isinstance(obj, partial):
            keywords = self._serialize(obj.keywords)
            return {
                'name': obj.func.__name__,
                'params':keywords
            }
        # Constant
        elif isinstance(obj, Enum):
            return obj.value
        # Slice
        elif isinstance(obj, slice):
            if obj.start is None and obj.stop is None and obj.step is None:
                return '[:]'
            parts = []
            parts.append(str(obj.start) if obj.start is not None else '')
            parts.append(str(obj.stop) if obj.stop is not None else '')
            if obj.step is not None:
                parts.append(str(obj.step))
            return f"[{':'.join(parts)}]"
        elif isinstance(obj, torch.device):
            return str(obj)
        # Optimizer
        elif isinstance(obj, torch.optim.Optimizer):
            param_groups = obj.state_dict()['param_groups'].copy()
            for group in param_groups:
                group.pop('params')
                for key, default_value in obj.defaults.items():
                    if group[key] == default_value:
                        group.pop(key)
            return {
                'name': obj.__class__.__name__,
                'params': param_groups
            }
        # Other objects
        else:
            sig = inspect.signature(obj.__class__.__init__)
            params = {}
            for k, v in vars(obj).items():
                if k in sig.parameters:
                    is_default = v == sig.parameters[k].default
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        is_default = bool(is_default.all())
                    if not isinstance(is_default, bool):
                        raise ValueError(f'Parameter {k} of {obj.__class__.__name__} is not serializable.')
                    if not is_default:
                        params[k] = repr(v)

            return {
                'name': obj.__class__.__name__,
                'params': params
            }

    def _get_md5(self, serialized: dict) -> str:
        """Get MD5 value of training-dependent part of config."""
        td_cfg = copy.deepcopy(serialized)
        for k in self._TRAINING_INDEPENDENT_KEYS:
            if k in td_cfg:
                td_cfg.pop(k)
        td_str = json.dumps(td_cfg, ensure_ascii=False, indent=4).encode('utf-8')
        return hashlib.md5(td_str).hexdigest()
