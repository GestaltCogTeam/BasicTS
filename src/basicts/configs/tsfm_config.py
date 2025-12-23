from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW

from basicts.data import BasicTSForecastingDataset
from basicts.runners.callback import BasicTSCallback
from basicts.runners.optim.lr_schedulers import CosineWarmup
from basicts.runners.taskflow import (BasicTSForecastingTaskFlow,
                                      BasicTSTaskFlow)

from .base_config import BasicTSConfig
from .model_config import BasicTSModelConfig


@dataclass(init=False)
class BasicTSFoundationModelConfig(BasicTSConfig):

    """
    BasicTS Foundation Model Config, including general configuration, dataset and scaler configuration, model configuration, \
    metrics configuration, training configuration, validation configuration, test configuration, evaluation configuration, \
    and environment configuration.
    
    **Required Fields:** These fields must be specified for running BasicTS forecasting tasks.
    - `dataset_name` (str): Dataset name.
    - `model` (cls): Model class. You can pass its class name as string and it will be transformed into class type automatically.
    - `model_params` (EasyDict): Model parameters. You can pass it as dict and it will be transformed into EasyDict automatically.
    
    **Hot Fields:** Though these parameters have default settings, they are likely to be modified frequently.
    - `gpus` (str|None): The used GPU devices (e.g., '0,1,2,3'). Default: None (on CPU).
    - `callbacks` (List[BasicTSCallback]): Callbacks. Specific functions such as early stopping, gradient clipping can be added \
        to runner by callbacks. Default: [].
    - `num_epochs` or `num_steps` (int): Number of epochs or steps of training. Only one of them should be specified. \
        Default: `num_epochs=100` and `num_steps=None`.
    - `batch_size` (int): Batch size. If you specify this field, all dataloader will be setted to the same batch size. \
        You can also set them separately in `train_batch_size`, `val_batch_size`, and `test_batch_size`. Default: 64.
    - `loss` (cls): Loss function. You can pass it as a string in `basicts.metrics` module and it will be transformed \
        into cls automatically. Default: MAE.
    - `optimizer` (Optimizer): Optimizer type. Default: Adam.
    - `optimizer_params` (dict): Optimizer parameters. Default: {'lr': 0.0002, 'weight_decay': 0.0005}.
    - `lr_scheduler` (LRScheduler): Learning rate scheduler. Default: None.
    - `lr_scheduler_params` (dict): Learning rate scheduler parameters. Default: {}.
    - `seed` (int): Random seed. Default: 42.

    **BasicTSObject parameters:** You can two ways to construct a BasicTSObject () except for model.
    - Pass parameters as dict to the params field. For example:
      
      >>> config = BasicTSForecastingConfig(
      >>>     dataset_params={
      >>>         'input_len': 336,
      >>>         'output_len': 336,
      >>>         ...},
      >>>     ...
      >>> )
    - Pass each parameter as field. For example:

      >>> config = BasicTSFoundationModelConfig(
      >>>     input_len=336,
      >>>     output_len=336,
      >>>     ...
      >>> )
      When using your own dataset, you can pass extra parameters by:

      >>> config["your_param_key"] = your_param
    """

    ################################# Required Fields #################################

    model: type = field(metadata={"help": "Model class. Must be specified."})

    model_config: BasicTSModelConfig = field(metadata={"help": "Model configuration. Must be specified."})

    dataset_name: str = field(default=None, metadata={"help": "Dataset name. Must be specified if it is not in `dataset_params`."})

    ############################## General Configuration ##############################

    gpus: Union[str, None] = field(
        default=None,
        metadata={"help": "Wether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'."})

    gpu_num: int = field(default=0, metadata={"help": "Post-init. Number of GPUs."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    taskflow: BasicTSTaskFlow = field(default=BasicTSForecastingTaskFlow(),
                                      metadata={"help": "Taskflow."})

    callbacks: List[BasicTSCallback] = field(default_factory=list,
                                             metadata={"help": "Callbacks."})

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`."})

    compile_model: bool = field(default=False, metadata={"help": "Whether to compile model."})

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    dataset_type: type = field(default=BasicTSForecastingDataset, metadata={"help": "Dataset type."})
    dataset_params: dict = field(default_factory=dict)
    input_len: int = field(default=None, metadata={"help": "Input length."})
    output_len: int = field(default=None, metadata={"help": "Output length."})
    use_timestamps: bool = field(default=None, metadata={"help": "Whether to use timestamps as supplementary."})
    memmap: bool = field(default=None, metadata={"help": "Whether to use memmap to load datasets."})
    batch_size: Optional[int] = field(default=None, metadata={"help": "Batch size. If setted, all dataloaders will be setted to the same batch size."})
    null_val: float = field(default=np.nan, metadata={"help": "Null value."})
    null_to_num: float = field(default=0.0, metadata={"help": "Null value to number."})

    # Scaler settings
    scaler: type = None # Post-init. Scaler.
    norm_each_channel: bool = None # Post-init. Whether to normalize data for each channel independently.
    rescale: bool = False # Whether to rescale data. Default: False

    ############################## Model Configuration ##############################

    # Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`.
    # In distributed computing, if there are unused parameters in the forward process, PyTorch usually raises a RuntimeError.
    # In such cases, this parameter should be set to True.
    model_dtype: Union[torch.dtype, str] = "bfloat16"

    ############################## Metrics Configuration ##############################

    # Metrics settings
    metrics: List[Union[str, Tuple[str, Callable]]] = field(
        default_factory=list,
        metadata={"help": "Metric names. If metric is a string, it should be in `basicts.metrics.ALL_METRICS`. " \
                  "Otherwise, it should be a tuple of (metric_name: str, metric_function: Callable)."})
    target_metric: str = "loss" # Target metric, used for saving best checkpoints. It should be in `metrics` or a string "loss".
    best_metric: Literal["min", "max"] = "min" # Best metric, used for saving best checkpoints. 'min' or 'max'. Default: 'min'. If 'max', the larger the metric, the better.

    ############################## Training Configuration ##############################

    num_epochs: int = None
    num_steps: int = 10_000

    # Loss function
    loss: Union[str, Callable] = field(
        default="MAE", metadata={"help": "Loss function. If a string, it should be in `basicts.metrics.ALL_METRICS`."})

    # Optimizer
    optimizer: type = field(default=AdamW)
    optimizer_params: dict = field(default_factory=lambda: {"lr": 1e-3, "fused": True})
    lr: float = field(default=None, metadata={"help": "Learning rate."})

    # Learning rate scheduler
    lr_scheduler: type = field(default=CosineWarmup)
    lr_scheduler_params: dict = field(default_factory=lambda: {"num_warmup_steps": 1000, "num_training_steps": 10_000})

    # Train data loader settings
    train_batch_size: int = 32
    train_data_prefetch: bool = False # Whether to use dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator. Default: False.
    train_data_shuffle: bool = True # Whether to shuffle the training data. Default: False
    train_data_collate_fn: Callable = None # Collate function for the training dataloader. Default: None
    train_data_num_workers: int = 0 # Number of workers for the training dataloader. Default: 0
    train_data_pin_memory: bool = False # Whether to pin memory for the training dataloader. Default: False

    ############################## Validation Configuration ##############################

    val_batch_size: int = 32
    val_interval: int = 1000
    val_data_prefetch: bool = False
    val_data_shuffle: bool = False
    val_data_collate_fn: Callable = None
    val_data_num_workers: int = 0
    val_data_pin_memory: bool = False

    # Checkpoint loading and saving settings
    # Directory to save checkpoints. Default: 'checkpoints/{model}/{dataset}_{num_epochs}_{input_len}_{output_len}', which will be loaded lazily.
    ckpt_save_dir: str = None
    # Checkpoint save strategy. `CFG.TRAIN.CKPT_SAVE_STRATEGY` should be None, an int value, a list or a tuple. Default: None.
    # None: remove last checkpoint file every epoch.
    # Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch.
    # List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY, remove last checkpoint file when last_epoch not in ckpt_save_strategy
    ckpt_save_strategy: Union[int, list[int], tuple[int]] = val_interval
    finetune_from: str = None # Checkpoint path for fine-tuning. Default: None. If not specified, the model will be trained from scratch.
    strict_load: bool = True # Whether to strictly load the checkpoint. Default: True.
    eval_after_train: bool = field(default=False, metadata={"help": "Whether to evaluate the model after training."})
    save_results: bool = False # Whether to save evaluation results in a numpy file. Default: False

    ############################## Environment Configuration ##############################

    tf32: bool = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
    deterministic: bool = False # Whether to set the random seed to get deterministic results. Default: False
    cudnn_enabled: bool = True # Whether to enable cuDNN. Default: True
    cudnn_benchmark: bool = True# Whether to enable cuDNN benchmark. Default: True
    cudnn_determinstic: bool = False # Whether to set cuDNN to deterministic mode. Default: False

    ##################################### Post Init #######################################

    def __post_init__(self):
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f"checkpoints/{self.model.__class__.__name__}/{self.dataset_name}_{self.num_steps}"
