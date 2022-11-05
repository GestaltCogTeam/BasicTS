import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.archs import FEDformer
from basicts.runners import FEDformerRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "FEDformer model configuration"
CFG.RUNNER = FEDformerRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 96
CFG.DATASET_OUTPUT_LEN = 96
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
# CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "FEDformer"
CFG.MODEL.ARCH = FEDformer
NUM_NODES = 307
CFG.MODEL.PARAM = EasyDict(
    {
    "version": "Fourier",                       # for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]
    "mode_select": "random",                    # for FEDformer, there are two mode selection method, options: [random, low]
    "modes": 64,                                # modes to be selected random 64
    "seq_len": CFG.DATASET_INPUT_LEN,           # input sequence length
    "label_len": CFG.DATASET_INPUT_LEN/2,       # start token length used in decoder
    "pred_len": CFG.DATASET_OUTPUT_LEN,         # prediction sequence length\
    "output_attention": False,
    "embedding_type": "DataEmbedding",          # opt: DataEmbedding, DataEmbedding_wo_pos
    "moving_avg": 35,                           # window size of moving average
    "enc_in": NUM_NODES,                              # num nodes
    "dec_in": NUM_NODES,
    "d_model": 512,
    "num_time_features": 2,                     # number of used time features
    # "embed": "timeF",
    # "freq": "h",
    "dropout": 0.05,
    "base": "legendre",                         # mwt base
    "L": 3,                                     # ignore level
    "cross_activation": "tanh",                 # mwt cross atention activation function tanh or softmax
    "n_heads": 8,
    "d_ff": 2048,
    "activation": "gelu",
    "e_layers": 2,                              # num of encoder layers
    "c_out": NUM_NODES,
    "d_layers": 1                               # num of decoder layers
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0005,
    "weight_decay": 0.0005,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50],
    "gamma": 0.5
}

# ================= train ================= #
# CFG.TRAIN.CLIP_GRAD_PARAM = {
#     "max_norm": 5.0
# }
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
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
CFG.TEST.EVALUATION_HORIZONS = [12, 24, 48, 96]
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
