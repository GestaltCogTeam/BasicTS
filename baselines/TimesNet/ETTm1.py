import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mse, masked_mae

from .arch import TimesNet

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "TimesNet model configuration  "
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "ETTm1"
CFG.DATASET_TYPE = "Electricity Transformer Temperature"
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
CFG.MODEL.NAME = "TimesNet"
CFG.MODEL.ARCH = TimesNet
NUM_NODES = 7
CFG.MODEL.PARAM = {
    "seq_len": CFG.DATASET_INPUT_LEN,
    "label_len": CFG.DATASET_INPUT_LEN // 2,  # start token length used in decoder
    "pred_len": CFG.DATASET_OUTPUT_LEN,  # prediction sequence length
    "enc_in": NUM_NODES,                        # num nodes
    "c_out": NUM_NODES,
    "top_k": 5,                                # attn factor
    "d_model": 16,
    "d_ff": 32,
    "num_kernels": 6,
    "e_layers": 2,                              # num of encoder layers
    "embed": "timeF",  # [timeF, fixed, learned]
    "dropout": 0.05,
    "num_time_features": 4,                     # number of used time features
    "time_of_day_size": 24 * 4,
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366
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
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
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
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]
