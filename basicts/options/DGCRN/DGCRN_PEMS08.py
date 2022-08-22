import os
from easydict import EasyDict
import torch
# runner
from basicts.runners.DGCRN_runner import DGCRNRunner
from basicts.data.base_dataset import BaseDataset
from basicts.metrics.mae import masked_mae
from basicts.metrics.mape import masked_mape
from basicts.metrics.rmse import masked_rmse
from basicts.losses.losses import masked_l1_loss
from basicts.utils.serialization import load_adj

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = 'DGCRN model configuration'
CFG.RUNNER  = DGCRNRunner
CFG.DATASET_CLS   = BaseDataset
CFG.DATASET_NAME  = "PEMS08"
CFG.DATASET_TYPE  = 'Traffic flow'
CFG.GPU_NUM = 1
CFG.METRICS = {
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "MAPE": masked_mape
}

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED    = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME  = 'DGCRN'
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")
CFG.MODEL.PARAM = {
    "gcn_depth" : 2, 
    "num_nodes" : 170, 
    "predefined_A": [torch.Tensor(_) for _ in adj_mx], 
    "dropout"   : 0.3, 
    "subgraph_size" : 20, 
    "node_dim"  : 40, 
    "middle_dim": 2, 
    "seq_length": 12, 
    "in_dim"    : 2, 
    "list_weight": [0.05, 0.95, 0.95], 
    "tanhalpha" : 3, 
    "cl_decay_steps" : 4000, 
    "rnn_size" : 64, 
    "hyperGNN_dim" : 16
}

CFG.MODEL.FROWARD_FEATURES = [0, 1]            
CFG.MODEL.TARGET_FEATURES  = [0]                

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_l1_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.001,
    "weight_decay":0.0001
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[100, 150],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.NUM_EPOCHS    = 200
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA          = EasyDict()
CFG.TRAIN.NULL_VAL      = 0.0
## read data
CFG.TRAIN.DATA.DIR      = 'datasets/' + CFG.DATASET_NAME
## dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE   = 32
CFG.TRAIN.DATA.PREFETCH     = False
CFG.TRAIN.DATA.SHUFFLE      = True
CFG.TRAIN.DATA.NUM_WORKERS  = 2
CFG.TRAIN.DATA.PIN_MEMORY   = False
## curriculum learning
CFG.TRAIN.CL    = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS    = 0
CFG.TRAIN.CL.CL_EPOCHS      = 6
CFG.TRAIN.CL.PREDICTION_LENGTH  = 12

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
