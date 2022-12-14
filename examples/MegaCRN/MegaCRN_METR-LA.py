import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.archs import MegaCRN
from basicts.runners import MegaCRNRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils.serialization import load_pkl

from .loss import megacrn_loss


CFG = EasyDict()

# GTS does not allow to load parameters since it creates parameters in the first iteration
resume = False
if not resume:
    import random
    _ = random.randint(-1e6, 1e6)

# ================= general ================= #
CFG.DESCRIPTION = "MegaCRN model configuration"
CFG.RUNNER  = MegaCRNRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG._ = _
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 100
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MegaCRN"
CFG.MODEL.ARCH = MegaCRN

node_feats_full = load_pkl("datasets/{0}/data_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["processed_data"][..., 0]
train_index_list = load_pkl("datasets/{0}/index_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["train"]
node_feats = node_feats_full[:train_index_list[-1][-1], ...]
CFG.MODEL.PARAM = {
    "num_nodes": 207,
    "input_dim": 1,
    "output_dim": 1,
    "horizon": 12,
    "rnn_units": 32,
    "num_layers":1,
    "embed_dim":8,
    "cheb_k":3,
    "ycov_dim":1,
    "mem_num":10,
    "mem_dim":32,
    "memory_type":True,
    "meta_type":True,
    "decoder_type":'stepwise'
}
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = megacrn_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "eps": 1e-3
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.SETUP_GRAPH = True
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
# NOTE:
# TODO:
## WARNING: MegaCRN seems to fail when the train dataset is not shuffled.
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
