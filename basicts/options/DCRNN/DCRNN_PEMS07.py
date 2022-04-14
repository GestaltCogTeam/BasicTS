import os
from xmlrpc.client import FastParser
from easydict import EasyDict
import torch
# architecture 
from basicts.archs.DCRNN_arch import DCRNN
# runner
from basicts.runners.DCRNN_runner import DCRNNRunner
from basicts.data.base_dataset import BaseDataset
from basicts.metrics.mae import masked_mae
from basicts.metrics.mape import masked_mape
from basicts.metrics.rmse import masked_rmse
from basicts.losses.losses import maksed_l1_loss
from basicts.utils.serialization import load_adj

"""
We temporarily use the configs of PEMS04 without fine-tune. 
"""

CFG = EasyDict()

resume = False      # DCRNN does not allow to load parameters since it creates parameters in the first iteration
if not resume:
    import random
    _ = random.randint(-1e6, 1e6)

# ================= general ================= #
CFG.DESCRIPTION = 'DCRNN model configuration'
CFG.RUNNER  = DCRNNRunner
CFG.DATASET_CLS   = BaseDataset
CFG.DATASET_NAME  = "PEMS07"
CFG.DATASET_TYPE  = 'Traffic speed'
CFG.GPU_NUM = 1
CFG.SEED    = 1
CFG._       = _
CFG.CUDNN_ENABLED = True
CFG.METRICS = {
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "MAPE": masked_mape
}

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME  = 'DCRNN'
CFG.MODEL.ARCH  = DCRNN
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")
CFG.MODEL.PARAM = {
    "cl_decay_steps"    : 2000,
    "horizon"           : 12,
    "input_dim"         : 2,
    "max_diffusion_step": 2,
    "num_nodes"         : 883,
    "num_rnn_layers"    : 2,
    "output_dim"        : 1,
    "rnn_units"         : 64,
    "seq_len"           : 12,
    "adj_mx"            : [torch.tensor(i).cuda() for i in adj_mx],
    "use_curriculum_learning": True
}
CFG.MODEL.FROWARD_FEATURES = [0, 1]            # traffic speed, time in day
CFG.MODEL.TARGET_FEATURES  = [0]                # traffic speed

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = maksed_l1_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.003,
    "eps":1e-3
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[80],
    "gamma":0.3
}

# ================= train ================= #
# CFG.TRAIN.CLIP          = 5
CFG.TRAIN.NUM_EPOCHS    = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.SETUP_GRAPH   = True
# train data
CFG.TRAIN.DATA          = EasyDict()
CFG.TRAIN.NULL_VAL      = 0.0
## read data
CFG.TRAIN.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE   = 64
CFG.TRAIN.DATA.PREFETCH     = False
CFG.TRAIN.DATA.SHUFFLE      = True
CFG.TRAIN.DATA.NUM_WORKERS  = 2
CFG.TRAIN.DATA.PIN_MEMORY   = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
## read data
CFG.VAL.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE     = 64
CFG.VAL.DATA.PREFETCH       = False
CFG.VAL.DATA.SHUFFLE        = False
CFG.VAL.DATA.NUM_WORKERS    = 2
CFG.VAL.DATA.PIN_MEMORY     = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# validating data
CFG.TEST.DATA = EasyDict()
## read data
CFG.TEST.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE    = 64
CFG.TEST.DATA.PREFETCH      = False
CFG.TEST.DATA.SHUFFLE       = False
CFG.TEST.DATA.NUM_WORKERS   = 2
CFG.TEST.DATA.PIN_MEMORY    = False
