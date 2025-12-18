from dataclasses import dataclass, field
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
from torch.optim import Adam

from basicts.data import BasicTSImputationDataset
from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import BasicTSImputationTaskFlow, BasicTSTaskFlow
from basicts.scaler import ZScoreScaler

from .base_config import BasicTSConfig
from .model_config import BasicTSModelConfig


@dataclass(init=False)
class BasicTSImputationConfig(BasicTSConfig):

    """
    BasicTS Imputation Config, including general configuration, dataset and scaler configuration, model configuration, \
    metrics configuration, training configuration, validation configuration, test configuration, evaluation configuration, \
    and environment configuration.
    
    **Required Fields:** These fields must be specified for running BasicTS imputation tasks.
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
    - `save_results` (bool): Whether to save results. Default: False.

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

      >>> config = BasicTSForecastingConfig(
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

    taskflow: BasicTSTaskFlow = field(default=BasicTSImputationTaskFlow(),
                                      metadata={"help": "Taskflow."})

    callbacks: List[BasicTSCallback] = field(default_factory=list,
                                             metadata={"help": "Callbacks."})

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`."})

    compile_model: bool = field(default=False, metadata={"help": "Whether to compile model."})

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    dataset_type: type = field(default=BasicTSImputationDataset, metadata={"help": "Dataset type."})
    dataset_params: Union[dict, None] = field(
        default_factory=lambda: {
            "input_len": 336,
            "use_timestamps": True,
            "memmap": False,
        }, metadata={"help": "Dataset parameters."})

    # shortcuts
    input_len: int = field(default=None, metadata={"help": "Input length."})
    use_timestamps: bool = field(default=None, metadata={"help": "Whether to use timestamps as supplementary."})
    memmap: bool = field(default=None, metadata={"help": "Whether to use memmap to load datasets."})
    batch_size: Union[int, None] = field(
        default=None, metadata={"help": "Batch size. If setted, all dataloaders will be setted to the same batch size."})

    mask_ratio: float = field(default=0.25, metadata={"help": "Mask ratio."})
    null_val: float = field(default=np.nan, metadata={"help": "Null value."})
    null_to_num: float = field(default=0.0, metadata={"help": "Null value to number."})

    # Scaler settings
    scaler: type = field(default=ZScoreScaler, metadata={"help": "Scaler type."})
    norm_each_channel: bool = field(default=True, metadata={"help": "Whether to normalize data for each channel independently."})
    rescale: bool = field(default=False, metadata={"help": "Whether to rescale data."})

    ############################## Metrics Configuration ##############################

    metrics: List[Union[str, Tuple[str, Callable]]] = field(
        default_factory=lambda: ["MAE", "MSE", "RMSE", "MAPE", "WAPE"],
        metadata={"help": "Metric names. If metric is a string, it should be in `basicts.metrics.ALL_METRICS`. " \
                  "Otherwise, it should be a tuple of (metric_name: str, metric_function: Callable)."})

    target_metric: str = field(
        default="MAE",
        metadata={"help": "Target metric, used for saving best checkpoints. It should be in `metrics` or a string \"loss\"."})

    best_metric: Literal["min", "max"] = field(
        default="min",
        metadata={"help": "Best metric, used for saving best checkpoints." \
                  "Should be \"min\" or \"max\". If \"max\", the larger the metric, the better."})

    ############################## Training Configuration ##############################

    num_epochs: int = field(
        default=100, metadata={"help": "Number of epochs. If not None, the training will stop after `num_epochs` epochs."})

    num_steps: Union[int, None] = field(
        default=None, metadata={"help": "Number of steps. If not None, the training will stop after `num_steps` steps."})

    loss: Union[str, Callable] = field(
        default="MAE", metadata={"help": "Loss function. If a string, it should be in `basicts.metrics.ALL_METRICS`."})

    # Optimizer
    optimizer: type = field(default=Adam, metadata={"help": "Optimizer class."})
    optimizer_params: dict = field(
        default_factory=lambda: {"lr": 2e-4, "weight_decay": 5e-4},
        metadata={"help": "Optimizer parameters."})
    lr: float = field(default=None, metadata={"help": "Learning rate."})

    # Learning rate scheduler
    lr_scheduler: Union[type, None] = field(default=None, metadata={"help": "Learning rate scheduler type."})
    lr_scheduler_params: Union[dict, None] = field(default=None, metadata={"help": "Learning rate scheduler parameters."})

    # Checkpoint loading and saving settings
    ckpt_save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save checkpoints." \
                  "Default: 'checkpoints/{model_name}/{dataset_name}_{num_epochs}_{input_len}_{mask_ratio}', which will be post-initialized."})

    ckpt_save_strategy: Union[int, List[int], Tuple[int]] = field(
        default_factory=lambda: None,
        metadata={"help": "Checkpoint save strategy. " \
                  "None: remove last checkpoint file every epoch. " \
                  "Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch. " \
                  "List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY`, " \
                    "remove last checkpoint file when last_epoch not in ckpt_save_strategy."})

    finetune_from: Union[str, None] = field(
        default=None,
        metadata={"help": "Checkpoint path for fine-tuning. If not specified, the model will be trained from scratch."})

    strict_load: bool = field(default=True, metadata={"help": "Whether to strictly load the checkpoint."})

    # Train data loader settings
    train_batch_size: int = field(
        default=64, metadata={"help": "Batch size for training."})
    train_data_prefetch: bool = field(
        default=False, metadata={"help": "Whether to use dataloader with prefetch." \
                                 "See https://github.com/justheuristic/prefetch_generator."})
    train_data_shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the training data."})
    train_data_collate_fn: Union[Callable, None] = field(
        default=None, metadata={"help": "Collate function for the training dataloader."})
    train_data_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for the training dataloader."})
    train_data_pin_memory: bool = field(
        default=False, metadata={"help": "Whether to pin memory for the training dataloader."})

    ############################## Validation Configuration ##############################

    val_batch_size: int = field(
        default=64, metadata={"help": "Batch size for validation."})
    val_interval: int = field(
        default=1, metadata={"help": "Conduct validation every `val_interval` epochs."})
    val_data_prefetch: bool = field(
        default=False, metadata={"help": "Whether to use dataloader with prefetch." \
                                 "See https://github.com/justheuristic/prefetch_generator."})
    val_data_shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle the validation data."})
    val_data_collate_fn: Union[Callable, None] = field(
        default=None, metadata={"help": "Collate function for the validation dataloader."})
    val_data_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for the validation dataloader."})
    val_data_pin_memory: bool = field(
        default=False, metadata={"help": "Whether to pin memory for the validation dataloader."})

    ############################## Test Configuration ##############################

    test_batch_size: int = field(
        default=64, metadata={"help": "Batch size for testing."})
    test_interval: int = field(
        default=1, metadata={"help": "Conduct testing every `test_interval` epochs."})
    test_data_prefetch: bool = field(
        default=False, metadata={"help": "Whether to use dataloader with prefetch." \
                                 "See https://github.com/justheuristic/prefetch_generator."})
    test_data_shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle the testing data."})
    test_data_collate_fn: Union[Callable, None] = field(
        default=None, metadata={"help": "Collate function for the testing dataloader."})
    test_data_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for the testing dataloader."})
    test_data_pin_memory: bool = field(
        default=False, metadata={"help": "Whether to pin memory for the testing dataloader."})

    ########################### Evaluation Configuration ##########################

    eval_after_train: bool = field(
        default=True, metadata={"help": "Whether to evaluate the model after training."})

    save_results: bool = field(
        default=False, metadata={"help": "Whether to save evaluation results in a numpy file."})

    ############################## Environment Configuration ##############################

    tf32: bool = field(
        default=False, metadata={"help": "Whether to use TensorFloat-32 in GPU." \
                                 "See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere."})

    deterministic: bool = field(
        default=False, metadata={"help": "Whether to set the random seed to get deterministic results."})

    cudnn_enabled: bool = field(
        default=True, metadata={"help": "Whether to enable cuDNN."})

    cudnn_benchmark: bool = field(
        default=True, metadata={"help": "Whether to enable cuDNN benchmark."})

    cudnn_determinstic: bool = field(
        default=False, metadata={"help": "Whether to set cuDNN to deterministic mode."})

    ##################################### Post Init #######################################

    def __post_init__(self):
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f"checkpoints/{self.model.__name__}/{self.dataset_name}_{self.num_epochs}_{self.input_len}_{self.mask_ratio}"
