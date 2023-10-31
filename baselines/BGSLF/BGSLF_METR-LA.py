import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils.serialization import load_pkl
from basicts.losses import masked_mae

from .arch import BGSLF

CFG = EasyDict()

# GTS does not allow to load parameters since it creates parameters in the first iteration
resume = False
if not resume:
    import random
    _ = random.randint(-1e6, 1e6)

# ================= general ================= #
CFG.DESCRIPTION = "BGSLF model configuration"
CFG.RUNNER  = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "METR-LA"
CFG.DATASET_TYPE = "Traffic speed"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG._ = _
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "BGSLF"
CFG.MODEL.ARCH = BGSLF
node_feats_full = load_pkl("datasets/{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN, CFG.get("RESCALE", True)))["processed_data"][..., 0]
train_index_list = load_pkl("datasets/{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN, CFG.get("RESCALE", True)))["train"]
node_feats = node_feats_full[:train_index_list[-1][-1], ...]
CFG.MODEL.PARAM = {
    "node_feas": torch.Tensor(node_feats),
    "temperature": 0.5,
    "args": EasyDict({
        "device": torch.device("cuda:0"),
        "cl_decay_steps": 2000,
        "filter_type": "dual_random_walk",
        "horizon": 12,
        "feas_dim": 1,
        "input_dim": 2,
        "ll_decay": 0,
        "num_nodes": 207,
        "max_diffusion_step": 2,
        "num_rnn_layers": 1,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": 12,
        "use_curriculum_learning": True,
        "embedding_size": 256,
        "kernel_size": 12,
        "freq": 288,
        "requires_graph": 2
    })

}
CFG.MODEL.SETUP_GRAPH = True
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.003,
    "eps": 1e-3
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 40],
    "gamma": 0.1
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
