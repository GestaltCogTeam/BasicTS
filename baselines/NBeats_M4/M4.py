import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import M4ForecastingDataset
from basicts.runners import M4ForecastingRunner

from .arch import NBeats

def get_cfg(seasonal_pattern):
    assert seasonal_pattern in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    prediction_len = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}[seasonal_pattern]
    num_nodes = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000, "Weekly": 359, "Daily": 4227, "Hourly": 414}[seasonal_pattern]
    history_size = 2
    history_len = history_size * prediction_len

    CFG = EasyDict()

    # ================= general ================= #
    CFG.DESCRIPTION = "Multi-layer perceptron model configuration "
    CFG.RUNNER = M4ForecastingRunner
    CFG.DATASET_CLS = M4ForecastingDataset
    CFG.DATASET_NAME = "M4_" + seasonal_pattern
    CFG.DATASET_INPUT_LEN = history_len
    CFG.DATASET_OUTPUT_LEN = prediction_len
    CFG.GPU_NUM = 1

    # ================= environment ================= #
    CFG.ENV = EasyDict()
    CFG.ENV.SEED = 1
    CFG.ENV.CUDNN = EasyDict()
    CFG.ENV.CUDNN.ENABLED = True

    # ================= model ================= #
    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = "NBeats"
    CFG.MODEL.ARCH = NBeats
    CFG.MODEL.PARAM = {
        "type": "generic",
        "input_size": CFG.DATASET_INPUT_LEN,
        "output_size": CFG.DATASET_OUTPUT_LEN,
        "layer_size": 512,
        "layers": 4,
        "stacks": 30
    }
    # CFG.MODEL.PARAM = {
    #     "type": "interpretable",
    #     "input_size": CFG.DATASET_INPUT_LEN,
    #     "output_size": CFG.DATASET_OUTPUT_LEN,
    #     "seasonality_layer_size": 2048,
    #     "seasonality_blocks": 3,
    #     "seasonality_layers": 4,
    #     "trend_layer_size": 256,
    #     "degree_of_polynomial": 2,
    #     "trend_blocks": 3,
    #     "trend_layers": 4,
    #     "num_of_harmonics": 1
    # }
    CFG.MODEL.FORWARD_FEATURES = [0]
    CFG.MODEL.TARGET_FEATURES = [0]

    # ================= optim ================= #
    CFG.TRAIN = EasyDict()
    CFG.TRAIN.LOSS = masked_mae
    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = "Adam"
    CFG.TRAIN.OPTIM.PARAM = {
        "lr": 0.002,
        "weight_decay": 0.0001,
    }
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    CFG.TRAIN.LR_SCHEDULER.PARAM = {
        "milestones": [1, 50, 80],
        "gamma": 0.5
    }

    # ================= train ================= #
    CFG.TRAIN.CLIP_GRAD_PARAM = {
        'max_norm': 5.0
    }
    CFG.TRAIN.NUM_EPOCHS = 52
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

    # ================= test ================= #
    CFG.TEST = EasyDict()
    CFG.TEST.INTERVAL = 52
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
    CFG.EVAL.HORIZONS = []
    CFG.EVAL.SAVE_PATH = os.path.abspath(__file__ + "/..")

    return CFG
