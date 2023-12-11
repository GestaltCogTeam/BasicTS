import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.runners import M4Runner
from basicts.losses import masked_mae
from basicts.data import M4ForecastingDataset
from basicts.utils import partial

from .mlp_arch import MultiLayerPerceptron
from .MASE import mase

seasonal_pattern = "Monthly"

frequence = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Weekly": 1, "Daily": 1, "Hourly": 24}[seasonal_pattern]
lookback_size = 2 # [2, 3, 4, 5, 6, 7]
CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "Multi-layer perceptron model configuration"
CFG.RUNNER = M4Runner
CFG.DATASET_CLS = M4ForecastingDataset
CFG.DATASET_NAME = "M4_" + seasonal_pattern
CFG.DATASET_OUTPUT_LEN = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}[seasonal_pattern]
CFG.DATASET_INPUT_LEN = lookback_size * CFG.DATASET_OUTPUT_LEN
CFG.GPU_NUM = 1
CFG.METRICS = {"MASE": partial(mase, frequency=frequence)}
CFG.TARGET_METRICS = "MASE"

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MultiLayerPerceptron"
CFG.MODEL.ARCH = MultiLayerPerceptron
CFG.MODEL.PARAM = {
    "history_seq_len": CFG.DATASET_INPUT_LEN,
    "prediction_seq_len": CFG.DATASET_OUTPUT_LEN,
    "hidden_dim": 32
}
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 30, 38, 46, 54, 62, 70, 80],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
