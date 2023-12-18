import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mse, masked_mae

from .arch import SciNet

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "SciNet model configuration "
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Weather"
CFG.DATASET_TYPE = "Weather Data"
CFG.DATASET_INPUT_LEN = 336
CFG.DATASET_OUTPUT_LEN = 336
CFG.GPU_NUM = 1
# CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "SciNet"
CFG.MODEL.ARCH = SciNet
NUM_NODES = 21
CFG.MODEL.PARAM =  {
    "input_len": CFG.DATASET_INPUT_LEN,
    "output_len": CFG.DATASET_OUTPUT_LEN,  # prediction sequence length
    "input_dim": NUM_NODES,                        # num nodes
    "hid_size": 1,                                # attn factor
    "num_stacks": 2,
    "num_levels": 3,
    "num_decoder_layer": 1,
    "concat_len": 0,
    "groups": 1,                              # num of encoder layers
    "kernel": 5,
    "dropout": 0.5,
    "single_step_output_One": False,
    "positionalE": False,
    "modified": True,
    "RIN": True
    }

CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0002,
    "weight_decay": 0.0005,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 25, 50],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
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
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = 'datasets/' + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]
