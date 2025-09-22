# 采样概率变化

import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler.z_score_scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_mse
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch import MOIRAI
from ...runner import MOIRAIEvalRunner


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

CONTEXT_LENGTH = None
PREDICTION_LENGTH = None

MODEL_ARCH = MOIRAI

MODEL_PARAM = {
    'prediction_length': PREDICTION_LENGTH,
    'context_length': CONTEXT_LENGTH,
    'patch_size': 32,
    'from_pretrained': "/path/to/your/model", # model id
    # 'from_pretrained': "/home/S22/workspace/BLAST_CKPTS/MOIRAI-Base-BLAST",
}
DATA_NAME = "Weather"

NUM_ITERATIONS = None

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'MOIRAI Base'
CFG.DEVICE = 'gpu'
CFG.DEVICE_NUM = 8
# Runner
CFG.RUNNER = MOIRAIEvalRunner

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.DTYPE= 'bfloat16'

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = True
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
    'normalize': True,
    'reduce': 'median'
})

regular_settings = get_regular_settings(dataset_name=DATA_NAME)
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': regular_settings['TRAIN_VAL_TEST_RATIO'],
    'input_len': CONTEXT_LENGTH,
    'output_len': PREDICTION_LENGTH,
    'overlap': True
})
CFG.TEST = EasyDict()
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 8
CFG.TEST.DATA.SHUFFLE = False

############################## TEST Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': regular_settings['TRAIN_VAL_TEST_RATIO'][0],
    'norm_each_channel': regular_settings['NORM_EACH_CHANNEL'],
    'rescale': regular_settings['RESCALE'],
})

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MSE': masked_mse,
                            })
CFG.METRICS.NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
