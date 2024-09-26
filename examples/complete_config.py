############################## Import Dependencies ##############################

import os
import sys
from easydict import EasyDict

# TODO: Remove this when basicts can be installed via pip
sys.path.append(os.path.abspath(__file__ + '/../../..'))

# Import metrics & loss functions
from basicts.metrics import masked_mae, masked_mape, masked_rmse
# Import dataset class
from basicts.data import TimeSeriesForecastingDataset
# Import runner class
from basicts.runners import SimpleTimeSeriesForecastingRunner
# Import scaler class
from basicts.scaler import ZScoreScaler
# Import model architecture
from .arch import MultiLayerPerceptron as MLP

from basicts.utils import get_regular_settings

############################## Hot Parameters ##############################

# Dataset & Metrics configuration
DATA_NAME = 'PEMS08'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

# Model architecture and parameters
MODEL_ARCH = MLP
MODEL_PARAM = {
    'history_seq_len': INPUT_LEN,
    'prediction_seq_len': OUTPUT_LEN,
    'hidden_dim': 64
}
NUM_EPOCHS = 100

############################## General Configuration ##############################

CFG = EasyDict()

# General settings
CFG.DESCRIPTION = 'An Example Config' # Description of this config, not used in the BasicTS, but useful for the user to remember the purpose of this config
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)

# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner # Runner class

############################## Environment Configuration ##############################

CFG.ENV = EasyDict() # Environment settings. Default: None

# GPU and random seed settings
CFG.ENV.TF32 = False # Whether to use TensorFloat-32 in GPU. Default: False. See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
CFG.ENV.SEED = 42 # Random seed. Default: None
CFG.ENV.DETERMINISTIC = False # Whether to set the random seed to get deterministic results. Default: False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True # Whether to enable cuDNN. Default: True
CFG.ENV.CUDNN.BENCHMARK = True# Whether to enable cuDNN benchmark. Default: True
CFG.ENV.CUDNN.DETERMINISTIC = False # Whether to set cuDNN to deterministic mode. Default: False

############################## Dataset Configuration ##############################

CFG.DATASET = EasyDict() # Dataset settings. Default: None. If not specified, get the training, validation, and test datasets from CFG.[TRAIN, VAL, TEST].DATA.DATASET.

# Dataset settings
CFG.DATASET.NAME = DATA_NAME # Name of the dataset, used for saving checkpoints and setting the process title.
CFG.DATASET.TYPE = TimeSeriesForecastingDataset # Dataset class use in both training, validation, and test.
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
}) # Parameters for the dataset class

############################## Scaler Configuration ##############################

CFG.SCALER = EasyDict() # Scaler settings. Default: None. If not specified, the data will not be normalized, i.e., the data will be used directly for training, validation, and test.

# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}) # Parameters for the scaler class

############################## Model Configuration ##############################

CFG.MODEL = EasyDict() # Model settings, must be specified.

# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__ # Model name, must be specified, used for saving checkpoints and set the process title.
CFG.MODEL.ARCH = MODEL_ARCH # Model architecture, must be specified.
CFG.MODEL.PARAM = MODEL_PARAM # Model parameters
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2] # Features used as input. The size of input data `history_data` is usually [B, L, N, C], this parameter specifies the index of the last dimension, i.e., history_data[:, :, :, CFG.MODEL.FORWARD_FEATURES].
CFG.MODEL.TARGET_FEATURES = [0] # Features used as output. The size of target data `future_data` is usually [B, L, N, C], this parameter specifies the index of the last dimension, i.e., future_data[:, :, :, CFG.MODEL.TARGET_FEATURES].
CFG.MODEL.SETUP_GRAPH = False # Whether to set up the computation graph. Default: False. Implementation of many works (e.g., DCRNN, GTS) acts like TensorFlow, which creates parameters in the first feedforward process.
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = False # Controls the `find_unused_parameters parameter` of `torch.nn.parallel.DistributedDataParallel`. In distributed computing, if there are unused parameters in the forward process, PyTorch usually raises a RuntimeError. In such cases, this parameter should be set to True.

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict() # Metrics settings. Default: None. If not specified, the default metrics will be used.

# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            }) # Metrics functions, default: MAE, MSE, RMSE, MAPE, WAPE
CFG.METRICS.TARGET = 'MAE' # Target metric, used for saving best checkpoints.
CFG.METRICS.BEST = 'min' # Best metric, used for saving best checkpoints. 'min' or 'max'. Default: 'min'. If 'max', the larger the metric, the better.
CFG.METRICS.NULL_VAL = NULL_VAL # Null value for the metric. Default: np.nan

############################## Training Configuration ##############################

CFG.TRAIN = EasyDict() # Training settings, must be specified for training.

# Training parameters
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
) # Directory to save checkpoints. Default: 'checkpoints/{MODEL_NAME}/{DATASET_NAME}_{NUM_EPOCHS}_{INPUT_LEN}_{OUTPUT_LEN}'
CFG.TRAIN.CKPT_SAVE_STRATEGY = None # Checkpoint save strategy. `CFG.TRAIN.CKPT_SAVE_STRATEGY` should be None, an int value, a list or a tuple. None: remove last checkpoint file every epoch. Default: None. Int: save checkpoint every `CFG.TRAIN.CKPT_SAVE_STRATEGY` epoch.  List or Tuple: save checkpoint when epoch in `CFG.TRAIN.CKPT_SAVE_STRATEGY, remove last checkpoint file when last_epoch not in ckpt_save_strategy
CFG.TRAIN.FINETUNE_FROM = None # Checkpoint path for fine-tuning. Default: None. If not specified, the model will be trained from scratch.
CFG.TRAIN.STRICT_LOAD = True # Whether to strictly load the checkpoint. Default: True.

# Loss function
CFG.TRAIN.LOSS = masked_mae # Loss function, must be specified for training.

# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict() # Optimizer settings, must be specified for training.
CFG.TRAIN.OPTIM.TYPE = 'Adam' # Optimizer type, must be specified for training.
CFG.TRAIN.OPTIM.PARAM = {
                            'lr': 0.002,
                            'weight_decay': 0.0001,
                        } # Optimizer parameters

# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict() # Learning rate scheduler settings. Default: None. If not specified, the learning rate will not be adjusted during training.
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR' # Learning rate scheduler type.
CFG.TRAIN.LR_SCHEDULER.PARAM = {
                            'milestones': [1, 50, 80],
                            'gamma': 0.5
                        } # Learning rate scheduler parameters

# Early stopping
CFG.TRAIN.EARLY_STOPPING_PATIENCE = None # Early stopping patience. Default: None. If not specified, the early stopping will not be used.

# gradient clip settings
CFG.TRAIN.CLIP_GRAD_PARAM = None # Gradient clipping parameters (torch.nn.utils.clip_grad_norm_). Default: None. If not specified, the gradient clipping will not be used.

# Curriculum learning settings
CFG.TRAIN.CL = EasyDict() # Curriculum learning settings. Default: None. If not specified, the curriculum learning will not be used.
CFG.TRAIN.CL.CL_EPOCHS = 1 # Number of epochs for each curriculum learning stage, must be specified if CFG.TRAIN.CL is specified.
CFG.TRAIN.CL.WARM_EPOCHS = 0 # Number of warm-up epochs. Default: 0
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN # Total prediction length, must be specified if CFG.TRAIN.CL is specified.
CFG.TRAIN.CL.STEP_SIZE = 1 # Step size for the curriculum learning. Default: 1. The current prediction length will be increased by CFG.TRAIN.CL.STEP_SIZE in each stage.

# Train data loader settings
CFG.TRAIN.DATA = EasyDict() # Training dataloader settings, must be specified for training.
CFG.TRAIN.DATA.PREFETCH = False # Whether to use dataloader with prefetch. See https://github.com/justheuristic/prefetch_generator. Default: False.
CFG.TRAIN.DATA.BATCH_SIZE = 64 # Batch size for training. Default: 1
CFG.TRAIN.DATA.SHUFFLE = True # Whether to shuffle the training data. Default: False
CFG.TRAIN.DATA.COLLATE_FN = None # Collate function for the training dataloader. Default: None
CFG.TRAIN.DATA.NUM_WORKERS = 0 # Number of workers for the training dataloader. Default: 0
CFG.TRAIN.DATA.PIN_MEMORY = False # Whether to pin memory for the training dataloader. Default: False

############################## Validation Configuration ##############################

CFG.VAL = EasyDict()

# Validation parameters
CFG.VAL.INTERVAL = 1 # Conduct validation every `CFG.VAL.INTERVAL` epochs. Default: 1
CFG.VAL.DATA = EasyDict() # See CFG.TRAIN.DATA
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.COLLATE_FN = None
CFG.VAL.DATA.NUM_WORKERS = 0
CFG.VAL.DATA.PIN_MEMORY = False

############################## Test Configuration ##############################

CFG.TEST = EasyDict()

# Test parameters
CFG.TEST.INTERVAL = 1 # Conduct test every `CFG.TEST.INTERVAL` epochs. Default: 1
CFG.TEST.DATA = EasyDict() # See CFG.TRAIN.DATA
CFG.VAL.DATA.PREFETCH = False
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.COLLATE_FN = None
CFG.TEST.DATA.NUM_WORKERS = 0
CFG.TEST.DATA.PIN_MEMORY = False

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12]  # The prediction horizons for evaluation. Default value: []. NOTE: HORIZONS[i] refers to testing **on the i-th** time step, representing the loss for that specific time step. This is a common setting in spatiotemporal forecasting. For long-sequence predictions, it is recommended to keep HORIZONS set to the default value [] to avoid confusion.
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
