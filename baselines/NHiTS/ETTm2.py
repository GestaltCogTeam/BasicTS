import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.metrics import masked_mse, masked_mae

from .arch import NHiTS

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "NHiTS Config  "
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ETTm2"
CFG.DATASET_TYPE = "Electricity Transformer Temperature"
CFG.DATASET_INPUT_LEN = 1680
CFG.DATASET_OUTPUT_LEN = 336
CFG.GPU_NUM = 1
CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "NHiTS"
CFG.MODEL.ARCH = NHiTS
CFG.MODEL.PARAM = {
    "context_length": CFG.DATASET_INPUT_LEN,
    "prediction_length": CFG.DATASET_OUTPUT_LEN,
    "output_size": 1,
    "n_blocks": [1, 1, 1],
    "n_layers": [2, 2, 2, 2, 2, 2, 2, 2],
    "hidden_size": [[512, 512], [512, 512], [512, 512]],
    "pooling_sizes": [8, 8, 8],
    "downsample_frequencies": [24, 12, 1]
}
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[20, 40, 60, 80],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "./checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = os.path.join("./datasets", CFG.DATASET_NAME)
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 128
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = os.path.join("./datasets", CFG.DATASET_NAME)
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 128
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = os.path.join("./datasets", CFG.DATASET_NAME)
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 128
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [6, 12, 24]
