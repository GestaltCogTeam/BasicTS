from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from basicts.data import BasicTSForecastingDataset
from basicts.metrics import masked_mae
from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import (BasicTSForecastingTaskFlow,
                                      BasicTSTaskFlow)
from basicts.scaler import BasicTSScaler, ZScoreScaler

from .base_config import BasicTSConfig


@dataclass
class BasicTSForecastingConfig(BasicTSConfig):

    """
    BasicTS Forecasting Config, including general configuration, dataset and scaler configuration, model configuration, \
    metrics configuration, training configuration, validation configuration, test configuration, evaluation configuration, \
    and environment configuration.
    
    **Required Field:** Decorated by `NotEmptyField`
    - `dataset_name` (str): Dataset name.
    - `model` (cls): Model class. You can pass its class name as string and it will be transformed into class type automatically.
    - `model_params` (EasyDict): Model parameters. You can pass it as dict and it will be transformed into EasyDict automatically.

    **Lazy Field:** Decorated by `LazyField`. If not specified, these fields will be loaded lazily from regular setting files.
    - `input_len` (int): The input length of time series.
    - `output_len` (int): The output length of time series.
    - `train_val_test_ratio` (List[float]): The split ratio of the dataset.
    - `null_val` (float): The null value of the dataset.
    - `rescale` (bool): Whether to rescale data.
    - `norm_each_channel` (bool): Whether to normalize data for each channel independently.
    
    **Hot field:** Though these parameters have default settings, they are likely to be modified frequently.
    - `gpus` (str|None): The used GPU devices (e.g., '0,1,2,3'). Default: None (on CPU).
    - `num_epochs` (int): Number of epochs. Default: 100.
    - `batch_size` (int): Batch size. If you specify this field, all dataloader will be setted to the same batch size. \
        You can also set them separately in `train_batch_size`, `val_batch_size`, and `test_batch_size`. Default: 64.
    - `forward_features` (slice or List[int]): Using which forward features (in most cases, this equals to whether to \
        use timestamps). Default: [0], i.e., only using temporal features.
    - `loss` (cls): Loss function. You can pass it as a string in `basicts.metrics` module and it will be transformed \
        into cls automatically. Default: MAE.
    - `loss_args` (dict): Arguments for loss function, if needed. Default: {}.
    - `optimizer` (Optimizer): Optimizer type. Default: Adam.
    - `optimizer_params` (dict): Optimizer parameters. Default: {'lr': 0.0002, 'weight_decay': 0.0005}.
    - `patience` (int): Early stopping patience. Default: 5.
    - `seed` (int): Random seed. Default: 42.
    - `save_results` (bool): Whether to save results. Default: False.
    """

    model: torch.nn.Module
    dataset_name: str

    taskflow: BasicTSTaskFlow = field(default=BasicTSForecastingTaskFlow(),
                                      metadata={"help": "Taskflow."})
    callbacks: List[BasicTSCallback] = field(default_factory=list,
                                             metadata={"help": "Callbacks."})

    ############################## General Configuration ##############################

    # General settings
    gpus: Optional[str] = field(default=None,
                                 metadata={"help": "Wether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'."})
    gpu_num: int = field(default=None, metadata={"help": "Post-init. Number of GPUs."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    ddp_find_unused_parameters: bool = field(default=False,
                                             metadata={"help": "Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`."})
    compile_model: bool = field(default=False, metadata={"help": "Whether to compile model."})

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    dataset_type: type = field(default=BasicTSForecastingDataset, metadata={"help": "Dataset type."})
    dataset_params: dict = field(default=None, metadata={"help": "Dataset parameters."})
    input_len: int = field(default=336, metadata={"help": "Input length."})
    output_len: int = field(default=336, metadata={"help": "Output length."})
    use_timestamps: bool = field(default=True, metadata={"help": "Whether to use timestamps as supplementary."})
    memmap: bool = field(default=False, metadata={"help": "Whether to use memmap to load datasets."})

    batch_size: Optional[int] = field(default=None, metadata={"help": "Batch size. If setted, all dataloaders will be setted to the same batch size."})
    null_val: float = field(default=np.nan, metadata={"help": "Null value."})
    null_to_num: float = field(default=0.0, metadata={"help": "Null value to number."})

    # Scaler settings
    scaler: BasicTSScaler = field(default=ZScoreScaler, metadata={"help": "Scaler type."})
    norm_each_channel: bool = field(default=True, metadata={"help": "Whether to normalize data for each channel independently."})
    rescale: bool = field(default=False, metadata={"help": "Whether to rescale data."})

    ############################## Metrics Configuration ##############################

    metrics: List[str] = field(default_factory=lambda: ["MAE", "MSE", "RMSE", "MAPE", "WAPE"], metadata={"help": "Metric names."})
    target_metric: str = field(default="MAE",
                               metadata={"help": "Target metric, used for saving best checkpoints. It should be in `metrics` or a string \"loss\"."})
    best_metric: Literal["min", "max"] = field(default="min",
                                               metadata={"help": "Best metric, used for saving best checkpoints." \
                                               "Should be \"min\" or \"max\". If \"max\", the larger the metric, the better."})

    ############################## Training Configuration ##############################

    num_epochs: int = 100
    num_steps: int = None

    # Loss function
    loss: Callable = masked_mae # Loss function

    # Optimizer
    optimizer: Optimizer = field(default=Adam, metadata={"help": "Optimizer class."})
    optimizer_params: dict = field(default_factory=lambda: {"lr": 2e-4, "weight_decay": 5e-4},
                                   metadata={"help": "Optimizer parameters."})

    # Learning rate scheduler
    lr_scheduler: LRScheduler = field(default=MultiStepLR, metadata={"help": "Learning rate scheduler type."})
    lr_scheduler_params: dict = field(default_factory=lambda: {"milestones": [1, 50], "gamma": 0.5},
                                      metadata={"help": "Learning rate scheduler parameters."})

    # Checkpoint loading and saving settings
    # Directory to save checkpoints. Default: 'checkpoints/{model}/{dataset}_{num_epochs}_{input_len}_{output_len}', which will be loaded lazily.
    ckpt_save_dir: str = None
    # Checkpoint save strategy. `CFG.TRAIN.CKPT_SAVE_STRATEGY` should be None, an int value, a list or a tuple. Default: None.
    # None: remove last checkpoint file every epoch.
    # Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch.
    # List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY, remove last checkpoint file when last_epoch not in ckpt_save_strategy
    ckpt_save_strategy: Union[int, List[int], Tuple[int]] = field(default_factory=lambda: None)
    finetune_from: str = None # Checkpoint path for fine-tuning. Default: None. If not specified, the model will be trained from scratch.
    strict_load: bool = True # Whether to strictly load the checkpoint. Default: True.

    # Train data loader settings
    train_batch_size: int = 64
    train_data_prefetch: bool = False # Whether to use dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator. Default: False.
    train_data_shuffle: bool = True # Whether to shuffle the training data. Default: False
    train_data_collate_fn: Callable = None # Collate function for the training dataloader. Default: None
    train_data_num_workers: int = 0 # Number of workers for the training dataloader. Default: 0
    train_data_pin_memory: bool = False # Whether to pin memory for the training dataloader. Default: False

    ############################## Validation Configuration ##############################

    val_batch_size: int = 64
    val_interval: int = 1 # Conduct test every `val_interval` epochs. Default: 1
    val_data_prefetch: bool = False
    val_data_shuffle: bool = False
    val_data_collate_fn: Callable = None
    val_data_num_workers: int = 0
    val_data_pin_memory: bool = False

    ############################## Test Configuration ##############################

    test_batch_size: int = 64
    test_interval: int = 1 # Conduct test every `test_interval` epochs. Default: 1
    test_data_prefetch: bool = False
    test_data_shuffle: bool = False
    test_data_collate_fn: Callable = None
    test_data_num_workers: int = 0
    test_data_pin_memory: bool = False

    ########################### Evaluation Configuration ##########################

    # Evaluation parameters
    save_results: bool = False # Whether to save evaluation results in a numpy file. Default: False
    evaluation_horizons: List[int] = None

    ############################## Environment Configuration ##############################

    tf32: bool = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
    deterministic: bool = False # Whether to set the random seed to get deterministic results. Default: False
    cudnn_enabled: bool = True # Whether to enable cuDNN. Default: True
    cudnn_benchmark: bool = True# Whether to enable cuDNN benchmark. Default: True
    cudnn_determinstic: bool = False # Whether to set cuDNN to deterministic mode. Default: False

    ############################## Training-Independent Keys ##############################

    _TRAINING_INDEPENDENT_KEYS: List[str] = field(default_factory=lambda: \
        ["gpus", "memmap", "ddp_find_unused_parameters", "compile_model", "ckpt_save_strategy", \
         "train_data_prefetch", "train_data_num_workers", "train_data_pin_memory", \
         "val_batch_size", "val_interval", "val_data_prefetch", "val_data_num_workers", "val_data_pin_memory", \
         "test_batch_size", "test_interval", "test_data_prefetch", "test_data_num_workers", "test_data_pin_memory", \
         "save_results",])

    ##################################### Post Init #######################################

    def __post_init__(self):
        if self.batch_size is not None:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f"checkpoints/{self.model.__class__.__name__}/{self.dataset_name}_{self.num_epochs}_{self.input_len}_{self.output_len}"
        self.gpu_num = len(self.gpus.split(",")) if self.gpus else 0
