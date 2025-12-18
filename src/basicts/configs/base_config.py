import copy
import hashlib
import importlib
import inspect
import json
import os
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from functools import partial
from numbers import Number
from types import FunctionType
from typing import (TYPE_CHECKING, Callable, List, Literal, Optional, Tuple,
                    Union)

import numpy as np
import torch
from easydict import EasyDict
from torch.optim.lr_scheduler import LRScheduler

from .model_config import BasicTSModelConfig

# avoid circular imports
if TYPE_CHECKING:
    from basicts.runners.callback import BasicTSCallback
    from basicts.runners.taskflow import BasicTSTaskFlow



@dataclass(init=False)
class BasicTSConfig(EasyDict):

    """
    Base class for configuration.

    Args:
        d (dict, optional): Dictionary to initialize the configuration. Defaults to None.
    """

    model: type
    model_config: BasicTSModelConfig

    dataset_name: str
    taskflow: "BasicTSTaskFlow"
    callbacks: List["BasicTSCallback"]

    ############################## General Configuration ##############################

    # General settings
    gpus: Optional[str] # Wether to use GPUs. The default is None (on CPU). For example, "0,1" is using "cuda:0" and "cuda:1".
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
    metrics: List[Union[str, Tuple[str, Callable]]]
    target_metric: str
    best_metric: Literal["min", "max"]

    ############################## Training Configuration ##############################

    num_epochs: int
    num_steps: int

    # Loss function
    loss: Union[str, Callable] # Loss function

    # Optimizer
    optimizer: type
    optimizer_params: dict
    lr: float

    # Learning rate scheduler
    lr_scheduler: type
    lr_scheduler_params: dict

    # Checkpoint loading and saving settings

    # Directory to save checkpoints. Default: "checkpoints/{model}/{dataset}_{num_epochs}_{input_len}_{output_len}", which will be loaded lazily.
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

    eval_after_train: bool
    save_results: bool # Whether to save evaluation results in a numpy file. Default: False

    ############################## Environment Configuration ##############################

    tf32: bool # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
    deterministic: bool # Whether to set the random seed to get deterministic results. Default: False
    cudnn_enabled: bool # Whether to enable cuDNN. Default: True
    cudnn_benchmark: bool # Whether to enable cuDNN benchmark. Default: True
    cudnn_determinstic: bool # Whether to set cuDNN to deterministic mode. Default: False

    ############################## Training-Independent Keys ##############################

    _md5: str = field(default="")
    _serialized: dict = field(default_factory=dict)
    _SHORTCUT_KEYS: List[str] = field(default_factory=lambda: \
        ["batch_size", "input_len", "output_len", "use_timestamps", "memmap", "lr"])
    _TRAINING_INDEPENDENT_KEYS: List[str] = field(default_factory=lambda: \
        ["gpus", "memmap", "ddp_find_unused_parameters", "compile_model", "ckpt_save_strategy", \
         "train_data_prefetch", "train_data_num_workers", "train_data_pin_memory", \
         "val_batch_size", "val_interval", "val_data_prefetch", "val_data_num_workers", "val_data_pin_memory", \
         "test_batch_size", "test_interval", "test_data_prefetch", "test_data_num_workers", "test_data_pin_memory", \
         "save_results", "eval_after_train"])

    #######################################################################################

    @property
    def md5(self) -> str:
        if not self._md5:
            self._md5 = self._get_md5(self.serialized)
        return self._md5

    @property
    def serialized(self) -> dict:
        if not self._serialized:
            self._serialized = self._serialize()
        return self._serialized

    def __init__(self, **kwargs):

        field_names = {f.name for f in fields(self)}

        init_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in field_names}

        # handling default values / default_factory
        for f in fields(self):
            if f.name not in init_kwargs:
                if f.default_factory is not MISSING:
                    init_kwargs[f.name] = f.default_factory()
                elif f.default is not MISSING:
                    init_kwargs[f.name] = f.default

        # set attributes from dataclass
        for k, v in init_kwargs.items():
            setattr(self, k, v)

        # pass other kwargs to EasyDict
        super().__init__(kwargs)

        # sub-class post init
        self.__post_init__()

        if self.batch_size is not None:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.model_config = self._pack_params(self.model, self.model_config)
        self.dataset_params = self._pack_params(self.dataset_type, self.dataset_params)
        self.optimizer_params = self._pack_params(self.optimizer, self.optimizer_params)
        self.lr_scheduler_params = self._pack_params(self.lr_scheduler, self.lr_scheduler_params)

        self.gpu_num = len(self.gpus.split(",")) if self.gpus else 0


    def __str__(self) -> str:
        return json.dumps(self.serialized, ensure_ascii=False, indent=4)

    def __getitem__(self, key):
        try:
            # for properties
            return getattr(self, key)
        except AttributeError:
            # compatible with old version
            if isinstance(key, str) and key not in self:
                key = key.replace(".", "_").lower()
            return super().__getitem__(key)

    @classmethod
    def from_json(cls, json_file_path: str):
        """Load config from a json file.

        Args:
            json_file_path (str): json file path
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        config_dict = json.loads(json_str)
        for k, v in config_dict.items():
            config_dict[k] = cls._construct_obj(v)
        config = cls(**config_dict)
        return config

    def _serialize(self) -> dict:
        serialized = {}
        for k, v in self.items():
            # serialize only non-private attributes and non-shortcut keys
            if not (k.startswith("_") or k in self._SHORTCUT_KEYS):
                serialized[k] = self._serialize_obj(v)
        return serialized

    @classmethod
    def _construct_obj(cls, obj: object) -> object:
        # List, tuple: Recursively construct objects in the list or tuple.
        if isinstance(obj, list):
            return [cls._construct_obj(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(cls._construct_obj(v) for v in obj)
        elif isinstance(obj, dict):
            if set(obj.keys()) == {"name", "module"}:
                module = importlib.import_module(obj["module"])
                return getattr(module, obj["name"])
            elif set(obj.keys()) == {"name", "module", "params"}:
                module = importlib.import_module(obj["module"])
                params = cls._construct_obj(obj["params"])
                return getattr(module, obj["name"])(**params)
            else: # Recursively construct objects in the dict.
                return {k: cls._construct_obj(v) for k, v in obj.items()}
        return obj

    def _pack_params(self, obj: type, obj_params: Union[dict, None]) -> dict:
        """Pack params to the config and add the extra keys to `_keys_to_pop`.

        Args:
            params (dict): params
            obj_params (dict): params of the object
        """

        if obj is None:
            return obj_params
        if obj_params is None:
            obj_params = {}

        init_fn = obj_params.__init__ \
            if isinstance(obj_params, BasicTSModelConfig) else obj.__init__
        sig = inspect.signature(init_fn)
        for k in sig.parameters.keys():
            if k == "self":
                continue
            # `optimizer` in LRScheduler will be passed by builder
            elif issubclass(obj, LRScheduler) and k == "optimizer":
                continue
            # short cut has higher priority than params in config
            elif k in self and self[k] is not None:
                obj_params[k] = self[k]
        return obj_params

    def save(self):
        json_str = str(self)
        save_dir = os.path.join(self.ckpt_save_dir, self.md5)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "cfg.json"), "w", encoding="utf-8") as f:
            f.write(json_str)

    def _serialize_obj(self, obj: object) -> object:
        if isinstance(obj, (str, bool, Number, type(None))):
            return obj
        # List, tuple
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_obj(v) for v in obj]
        # Dict
        elif isinstance(obj, dict):
            return {k: self._serialize_obj(v) for k, v in obj.items()}
        # Class or function
        elif isinstance(obj, (type, FunctionType)):
            return {
                "name": obj.__name__,
                "module": obj.__module__,
            }
        # Partial function
        elif isinstance(obj, partial):
            keywords = self._serialize_obj(obj.keywords)
            return {
                "name": obj.func.__name__,
                "params":keywords
            }
        # Constant
        elif isinstance(obj, Enum):
            return obj.value
        # Slice
        elif isinstance(obj, slice):
            if obj.start is None and obj.stop is None and obj.step is None:
                return "[:]"
            parts = []
            parts.append(str(obj.start) if obj.start is not None else "")
            parts.append(str(obj.stop) if obj.stop is not None else "")
            if obj.step is not None:
                parts.append(str(obj.step))
            joint = ":".join(parts)
            return f"[{joint}]"
        elif isinstance(obj, torch.device):
            return str(obj)
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
                        raise ValueError(f"Parameter {k} of {obj.__class__.__name__} is not serializable.")
                    if not is_default:
                        params[k] = self._serialize_obj(v)

            return {
                "name": obj.__class__.__name__,
                "module": obj.__module__,
                "params": params
            }

    def _get_md5(self, serialized: dict) -> str:
        """Get MD5 value of training-dependent part of config."""
        td_cfg = copy.deepcopy(serialized)
        for k in self._TRAINING_INDEPENDENT_KEYS:
            if k in td_cfg:
                td_cfg.pop(k)
        td_str = json.dumps(td_cfg, ensure_ascii=False, indent=4).encode("utf-8")
        return hashlib.md5(td_str).hexdigest()
