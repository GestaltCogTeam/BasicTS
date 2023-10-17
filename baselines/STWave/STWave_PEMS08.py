import os
import sys
import numpy as np
import scipy.sparse as sp
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from basicts.archs import STWave
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import stwave_masked_mae
from basicts.utils import load_adj

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def loadGraph(adj_mx, hs, ls):
    graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
    sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    return adj_gat, graphwave


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STWave model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS08"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STWave"
CFG.MODEL.ARCH = STWave
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME +
                     "/adj_mx.pkl", "original")
adjgat, gwv = loadGraph(_, 128, 1)
CFG.MODEL.PARAM = {
    "input_dim": 1,
    "hidden_size": 128,
    "layers": 2,
    "seq_len": 12,
    "horizon": 12,
    "log_samples": 1,
    "adj_gat": adjgat,
    "graphwave": gwv,
    "time_in_day_size": 288,
    "day_in_week_size": 7,
    "wave_type": "coif1",
    "wave_levels": 2,
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = stwave_masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [80, 90, 95],
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
