import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler.z_score_scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_mse
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch import ChronosBolt
from ...data import BLASTDatasetWoMixUp
from ...runner import ChronosRunner
from ...loss import fake_loss


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = ChronosBolt

context_length = 1024
predict_length = 64 # ref: /workspace/S22/TSFM_LLaMA3/huggingface_ckpts/chronos-bolt-tiny/config.json

MODEL_PARAM = {
    "model_id": "./baselines/ChronosBolt/ckpt/chronos-bolt-base",
    "from_pretrained": False,
    "device_map": "cpu",
}
DATA_NAME = "BLAST"

NUM_ITERATIONS = 100_000 # 总轮数
VAL_ITERATION_INTERVAL = 10_000 # 每VAL_ITERATION_INTERVAL执行一次验证

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'Chronos-Bolt Base | Debug: Data'
CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = ChronosRunner

# CFG.ENV = EasyDict() # Environment settings. Default: None
# CFG.ENV.SEED = 2025 # Random seed. Default: None

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.DTYPE= 'bfloat16'

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({})

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_STRATEGY = VAL_ITERATION_INTERVAL * 1 # 保存策略，每VAL_ITERATION_INTERVAL * 5保存一次模型
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)
CFG.TRAIN.LOSS = fake_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-3,
    "fused": True
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': int(NUM_ITERATIONS / 100 * 10), # 10%的warmup启动比例
    'num_training_steps': NUM_ITERATIONS,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 1.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 128
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.GRAD_ACCUMULATION_STEPS = 1

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = VAL_ITERATION_INTERVAL
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = BLASTDatasetWoMixUp
CFG.DATASET.PARAM = EasyDict({
    'context_length': context_length,
    'target_length': predict_length,
    'num_valid_samples': 1000
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
CFG.TEST.DATA.BATCH_SIZE = int(32 / (CFG.TEST.DATASET.CONTEXT_LENGTH / 640))
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


############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
})