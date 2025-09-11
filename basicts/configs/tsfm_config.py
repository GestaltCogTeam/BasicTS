from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from basicts.data import BuiltinTSForecastingDataset
from basicts.metrics import masked_mae
from basicts.runners.callback import BasicTSCallback
from basicts.runners.optim.lr_schedulers import CosineWarmup
from basicts.runners.taskflow import (BasicTSForecastingTaskFlow,
                                      BasicTSTaskFlow)
from basicts.scaler import BasicTSScaler
from basicts.utils import BasicTSTask

from .base_config import BasicTSConfig


@dataclass
class BasicTSFoundationModelConfig(BasicTSConfig):

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
    
    **Hot Field:** Though these parameters have default settings, they are likely to be modified frequently.
    - `gpus` (str|None): The used GPU devices (e.g., '0,1,2,3'). Default: None (on CPU).
    - `num_epochs` (int): Number of epochs. Default: 100.
    - `batch_size` (int): Batch size. If you specify this field, all dataloader will be setted to the same batch size. \
        You can also set them separately in `train_batch_size`, `val_batch_size`, and `test_batch_size`. Default: 64.
    - `forward_features` (slice or List[int]): Using which forward features (in most cases, this equals to whether to \
        use timestamps). Default: [0], i.e., only using temporal features.
    - `loss` (cls): Loss function. You can pass it as a string in `basicts.metrics` module and it will be transformed \
        into cls automatically. Default: MAE.
    - `loss_args` (dict): Arguments for loss function, if needed. Default: {}.
    - `optimizer` (str): Optimizer type. Default: Adam.
    - `optimizer_params` (dict): Optimizer parameters. Default: {'lr': 0.0002, 'weight_decay': 0.0005}.
    - `patience` (int): Early stopping patience. Default: 5.
    - `seed` (int): Random seed. Default: 42.
    - `save_results` (bool): Whether to save results. Default: False.
    """

    model: type
    dataset: BuiltinTSForecastingDataset
    taskflow: BasicTSTaskFlow = BasicTSForecastingTaskFlow()
    callbacks: List[BasicTSCallback] = field(default_factory=list)

    ############################## General Configuration ##############################

    # General settings
    task_name: BasicTSTask = BasicTSTask.TIME_SERIES_FORECASTING
    gpus: Optional[str] = None # Wether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'.
    gpu_num: int = None # Post-init. Number of GPUs.
    seed: int = 42 # Random seed.

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    batch_size: Optional[int] = None # if setted, all dataloaders will be setted to the same batch size.
    null_val: float = np.nan
    null_to_num: float = 0.0

    # Scaler settings
    scaler: BasicTSScaler = None # Post-init. Scaler.
    norm_each_channel: bool = None # Post-init. Whether to normalize data for each channel independently.
    rescale: bool = False # Whether to rescale data. Default: False

    ############################## Model Configuration ##############################

    # Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`.
    # In distributed computing, if there are unused parameters in the forward process, PyTorch usually raises a RuntimeError.
    # In such cases, this parameter should be set to True.
    model_dtype: Union[torch.dtype, str] = 'bfloat16'
    ddp_find_unused_parameters: bool = False

    compile_model: bool = False

    ############################## Metrics Configuration ##############################

    # Metrics settings
    metrics: List[str] = field(default_factory=lambda: ['MAE', 'MSE', 'RMSE', 'MAPE', 'WAPE']) # Metrics functions, default: MAE, MSE, RMSE, MAPE, WAPE
    target_metric: str = 'MAE' # Target metric, used for saving best checkpoints. It should be in `metrics` or a string "loss".
    best_metric: Literal['min', 'max'] = 'min' # Best metric, used for saving best checkpoints. 'min' or 'max'. Default: 'min'. If 'max', the larger the metric, the better.

    ############################## Training Configuration ##############################

    num_epochs: int = None
    num_steps: int = 10_000

    # Loss function
    loss: Callable = masked_mae # Loss function

    # Optimizer
    optimizer: Optimizer = None

    # Learning rate scheduler
    lr_scheduler: LRScheduler = None

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
    save_results: bool = False # Whether to save evaluation results in a numpy file. Default: False

    ############################## Environment Configuration ##############################

    tf32: bool = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
    deterministic: bool = False # Whether to set the random seed to get deterministic results. Default: False
    cudnn_enabled: bool = True # Whether to enable cuDNN. Default: True
    cudnn_benchmark: bool = True# Whether to enable cuDNN benchmark. Default: True
    cudnn_determinstic: bool = False # Whether to set cuDNN to deterministic mode. Default: False

    ############################## Training-Independent Keys ##############################

    _TRAINING_INDEPENDENT_KEYS: List[str] = field(default_factory=lambda: \
        ['gpus', 'memmap', 'ddp_find_unused_parameters', 'compile_model', 'ckpt_save_strategy', \
         'train_data_prefetch', 'train_data_num_workers', 'train_data_pin_memory', \
         'val_batch_size', 'val_interval', 'val_data_prefetch', 'val_data_num_workers', 'val_data_pin_memory', \
         ])

    ##################################### Post Init #######################################

    def __post_init__(self):
        # if self.cl_prediction_length is None:
        #     self.cl_prediction_length = self.dataset.output_len
        if self.batch_size is not None:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f'checkpoints/{self.model.__class__.__name__}/{self.dataset.name}_{self.num_steps}'

        # Follow the default settings in spatial-temporal forecasting and time series forecasting tasks.
        # if self.norm_each_channel is None:
        #     if self.task_name == BASICTS_TASK.SPATIAL_TEMPORAL_FORECASTING:
        #         self.norm_each_channel = False
        #     else: # time series forecasting
        #         self.norm_each_channel = True

        # Post-init optimizer and lr scheduler if not specified
        if self.optimizer is None:
            self.optimizer = AdamW(
                params = self.model.parameters(),
                lr=1e-3,
                fused=True
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = CosineWarmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.num_steps / 10,
                num_training_steps=self.num_steps
            )
        gpu_num = len(self.gpus.split(',')) if self.gpus else 0
        if self.gpu_num is not None:
            if self.gpu_num != gpu_num:
                raise ValueError(f'gpu_num ({self.gpu_num}) is not equal to the number of gpus {self.gpus}.')
        else:
            self.gpu_num = gpu_num
