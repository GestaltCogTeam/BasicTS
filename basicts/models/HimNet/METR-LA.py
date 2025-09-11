import os
import sys
import numpy as np
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_huber
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import HimNet
from .runner import HimNetRunner

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'METR-LA'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 12  # Length of input sequence
OUTPUT_LEN = 12  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = HimNet
HIMNET_CONFIG = {
  'lr': 0.001,
  'eps': 0.001,
  'weight_decay': 0.0005,
  'milestones': [30, 40],
  'clip_grad': 5,
  'batch_size': 16,
  'max_epochs': 200,
  'early_stop': 20
}

NUM_EPOCHS = HIMNET_CONFIG['max_epochs']

MODEL_PARAM = {
    'num_nodes': 207,
    'input_dim': 3,
    'output_dim': 1,
    'out_steps': 12,
    'hidden_dim': 64,
    'num_layers': 1,
    'cheb_k': 2,
    'ycov_dim': 2,
    'tod_embedding_dim': 8,
    'dow_embedding_dim': 8,
    'node_embedding_dim': 16,
    'st_embedding_dim': 16,
    'tf_decay_steps': 6000,
    'use_teacher_forcing': True
}

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = HimNetRunner


############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr":max(HIMNET_CONFIG['lr']*CFG.GPU_NUM, HIMNET_CONFIG['lr']),
    "eps":HIMNET_CONFIG['eps'],
    "weight_decay":HIMNET_CONFIG['weight_decay'],
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'milestones': HIMNET_CONFIG['milestones'],
    'gamma': HIMNET_CONFIG.get('lr_decay_rate', 0.1),
    'verbose': False
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = HIMNET_CONFIG['batch_size']
CFG.TRAIN.DATA.SHUFFLE = True

CFG.TRAIN.EARLY_STOPPING_PATIENCE = HIMNET_CONFIG['early_stop']

CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': HIMNET_CONFIG.get('clip_grad', 5)  
}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
