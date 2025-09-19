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
    'context_length': None,
    'trust_remote_code': True,
}
DATA_NAME = "BLAST"

# N = 20_000_000
# batch size = 16*8
# 20_000_000 / 16 / 8 = 156250 iterations

NUM_ITERATIONS = 200_000 # 总轮数
VAL_ITERATION_INTERVAL = 5_000 # 每VAL_ITERATION_INTERVAL执行一次验证

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'TimeMoE Large'
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
CFG.TRAIN.CKPT_SAVE_STRATEGY = VAL_ITERATION_INTERVAL * 1 # 保存策略，每VAL_ITERATION_INTERVAL * 5保存一次模型
CFG.TRAIN.LOSS = fake_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-4,
    "betas": (0.9, 0.95),
    "fused": True,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': int(NUM_ITERATIONS / 100 * 1), # 10%的warmup启动比例
    'num_training_steps': NUM_ITERATIONS,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 1.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True # has to be False
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.GRAD_ACCUMULATION_STEPS = 1

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = VAL_ITERATION_INTERVAL
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()
# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = BLASTDatasetMixUp
CFG.DATASET.PARAM = EasyDict({
    'num_valid_samples': 1000
})

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
    'normalize': not pretrained
})

############################## Test Configuration ##############################
# Only for evaluation after training

CFG.TEST = EasyDict()
CFG.TEST.DATASET = EasyDict()
CFG.TEST.DATASET.NAME = 'ETTm1'
testdata_regular_settings = get_regular_settings(CFG.TEST.DATASET.NAME)
CFG.TEST.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.TEST.DATASET.CONTEXT_LENGTH = 512
CFG.TEST.DATASET.PREDICTION_LENGTH = 96
CFG.TEST.DATASET.PARAM = EasyDict({
    'dataset_name': CFG.TEST.DATASET.NAME,
    'train_val_test_ratio': testdata_regular_settings['TRAIN_VAL_TEST_RATIO'],
    'input_len': CFG.TEST.DATASET.CONTEXT_LENGTH,
    'output_len': CFG.TEST.DATASET.PREDICTION_LENGTH,
    'overlap': True,
    # 'mode' is automatically set by the runner
})
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = int(4 / (CFG.TEST.DATASET.CONTEXT_LENGTH / 640))
CFG.TEST.DATA.SHUFFLE = False

############################## TEST Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': CFG.TEST.DATASET.NAME,
    'train_ratio': testdata_regular_settings['TRAIN_VAL_TEST_RATIO'][0],
    'norm_each_channel': testdata_regular_settings['NORM_EACH_CHANNEL'],
    'rescale': testdata_regular_settings['RESCALE'],
})

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MSE': masked_mse,
                            })
CFG.METRICS.NULL_VAL = testdata_regular_settings['NULL_VAL'] # Null value in the data


