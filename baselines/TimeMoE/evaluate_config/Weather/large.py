# 采样概率变化

import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler.z_score_scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_mse
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch import TimeMoE
from ...data import BLASTDatasetMixUp
from ...runner import TimeMoERunner
from ...loss import fake_loss


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

pretrained = False  # Whether to use a pretrained model

MODEL_ARCH = TimeMoE

MODEL_PARAM = {
    'model_id': "baselines/TimeMoE/ckpt/TimeMoE-200M",
    'from_pretrained': pretrained,
    'trust_remote_code': True,
}

DATA_NAME = 'Weather'

CONTEXT_LENGTH = None
PREDICTION_LENGTH = None

NUM_ITERATIONS = None

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'TimeMoE Base'
CFG.DEVICE = 'gpu'
CFG.DEVICE_NUM = 8
# Runner
CFG.RUNNER = TimeMoERunner

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

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
regular_settings = get_regular_settings(CFG.DATASET.NAME)
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
CFG.TEST.DATA.BATCH_SIZE = 2
CFG.TEST.DATA.SHUFFLE = False

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
    'normalize': not pretrained # TimeMoE-BLAST and original TimeMoE are trained with different data normalization strategies
})

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
