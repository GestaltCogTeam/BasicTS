from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple, Union

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from basicts.data import BuiltinTSForecastingDataset
from basicts.metrics import masked_mae
from basicts.scaler import BasicTSScaler, ZScoreScaler

from .base_config import BasicTSConfig
from .constants import BASICTS_TASK


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

    ############################## General Configuration ##############################

    # General settings
    task_name: Union[BASICTS_TASK, str] = BASICTS_TASK.TIME_SERIES_FORECASTING
    _runner: type = None # Runner class name
    gpus: Optional[str] = None # Wether to use GPUs. The default is None (on CPU). For example, '0,1' is using 'cuda:0' and 'cuda:1'.
    gpu_num: int = None # Post-init. Number of GPUs.
    seed: int = 42 # Random seed.

    ############################## Dataset and Scaler Configuration ##############################

    # Dataset settings
    batch_size: Optional[int] = None # if setted, all dataloaders will be setted to the same batch size.

    # Scaler settings
    scaler: BasicTSScaler = None # Post-init. Scaler.
    norm_each_channel: bool = None # Post-init. Whether to normalize data for each channel independently.
    rescale: bool = False # Whether to rescale data. Default: False

    ############################## Model Configuration ##############################

    # Features used in forward pass. The shape of input data is usually [B, L, N, C], this parameter specifies the index of the last dimension,
    # i.e., inputs[:, :, :, CFG.MODEL.TARGET_FEATURES]. Default: input_data[..., 0]
    forward_features: Union[slice, List[int]] = field(default_factory=lambda: [0])
    # Features used as output. The shape of target data is usually [B, L, N, C], this parameter specifies the index of the last dimension,
    # i.e., target[:, :, :, CFG.MODEL.TARGET_FEATURES]. Default: future_data[..., 0]
    target_features: Union[slice, List[int]] = field(default_factory=lambda: [0])
    # The index of the time series to be predicted, default is all ([..., :]). This setting is particularly useful in a Multivariate-to-Univariate setup.
    # For example, if 7 time series are input and the last two need to be predicted, you can set `CFG.MODEL.TARGET_TIME_SERIES=[5, 6]` to achieve this.
    target_time_series: Union[slice, List[int]] = slice(None)
    # Whether to set up the computation graph. Default: False.
    # Implementation of many works (e.g., DCRNN, GTS) acts like TensorFlow, which creates parameters in the first feedforward process.
    setup_graph: bool = False
    # Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`.
    # In distributed computing, if there are unused parameters in the forward process, PyTorch usually raises a RuntimeError.
    # In such cases, this parameter should be set to True.
    ddp_find_unused_parameters: bool = False

    compile_model: bool = False

    ############################## Metrics Configuration ##############################

    # Metrics settings
    metrics: List[str] = field(default_factory=lambda: ['MAE', 'MSE', 'RMSE', 'MAPE', 'WAPE']) # Metrics functions, default: MAE, MSE, RMSE, MAPE, WAPE
    target_metric: str = 'MAE' # Target metric, used for saving best checkpoints. It should be in `metrics` or a string "loss".
    best_metric: Literal['min', 'max'] = 'min' # Best metric, used for saving best checkpoints. 'min' or 'max'. Default: 'min'. If 'max', the larger the metric, the better.

    ############################## Training Configuration ##############################

    num_epochs: int = 100

    # Loss function
    loss: Callable = masked_mae # Loss function

    # Optimizer
    optimizer: Optimizer = None

    # Learning rate scheduler
    lr_scheduler: LRScheduler = None

    # Early stopping
    patience: int = 5 # Early stopping patience. Default: 5.

    # Gradient clipping parameters (torch.nn.utils.clip_grad_norm_). Default: None.
    clip_grad_param: dict = field(default_factory=lambda: None) # If not specified, the gradient clipping will not be used.

    # Curriculum learning settings
    cl: dict = field(default_factory=lambda: None) # Curriculum learning settings. Default: None. If not specified, the curriculum learning will not be used.
    cl_epochs: int = 1 # Number of epochs for each curriculum learning stage, must be specified if cl is specified.
    warm_epochs: int = 0 # Number of warm-up epochs. Default: 0
    cl_prediction_length: int = None # cl_prediction_length. Total prediction length, must be specified if cl is specified.
    cl_step_size: int = 1 # Step size for the curriculum learning. Default: 1. The current prediction length will be increased by CFG.TRAIN.CL.STEP_SIZE in each stage.

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
    # The prediction horizons for evaluation. Default value: [].
    # NOTE: HORIZONS[i] refers to testing **on the i-th** time step, representing the loss for that specific time step.
    # This is a common setting in spatiotemporal forecasting. For long-sequence predictions, it is recommended to keep HORIZONS set to the default value [] to avoid confusion.
    eval_horizons: List[int] = field(default_factory=lambda: [])
    eval_on_gpu: bool = True # Whether to use GPU for evaluation. Default: True
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
         'test_batch_size', 'test_interval', 'test_data_prefetch', 'test_data_num_workers', 'test_data_pin_memory', \
         'eval_horizons', 'eval_on_gpu', 'save_results',])

    #################################### Properties #######################################

    @property
    def runner(self) -> type:
        if self._runner is None:
            from basicts.runners import \
                SimpleTimeSeriesForecastingRunner  # pylint: disable=import-outside-toplevel
            self._runner = SimpleTimeSeriesForecastingRunner
        return self._runner

    @runner.setter
    def runner(self, value: type):
        self._runner = value

    def __post_init__(self):
        if self.cl_prediction_length is None:
            self.cl_prediction_length = self.dataset.output_len
        if self.batch_size is not None:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size
        if self.ckpt_save_dir is None:
            self.ckpt_save_dir = \
                f'checkpoints/{self.model.__class__.__name__}/{self.dataset.name}_{self.num_epochs}_{self.dataset.input_len}_{self.dataset.output_len}'

        # Follow the default settings in spatial-temporal forecasting and time series forecasting tasks.
        if self.norm_each_channel is None:
            if self.task_name == BASICTS_TASK.SPATIAL_TEMPORAL_FORECASTING:
                self.norm_each_channel = False
            else: # time series forecasting
                self.norm_each_channel = True

        # Post-init scaler if not specified
        if self.scaler is None:
            self.scaler = ZScoreScaler(self.norm_each_channel, self.rescale)

        # Post-init optimizer and lr scheduler if not specified
        if self.optimizer is None:
            self.optimizer = Adam(
                params = self.model.parameters(),
                lr = 0.0002,
                weight_decay = 0.0005
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = MultiStepLR(
                optimizer = self.optimizer,
                milestones = [1, int(self.num_epochs / 2)],
                gamma = 0.5
            )
        gpu_num = len(self.gpus.split(',')) if self.gpus else 0
        if self.gpu_num is not None:
            if self.gpu_num != gpu_num:
                raise ValueError(f'gpu_num ({self.gpu_num}) is not equal to the number of gpus {self.gpus}.')
        else:
            self.gpu_num = gpu_num
