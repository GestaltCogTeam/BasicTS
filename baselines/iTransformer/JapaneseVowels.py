import os
import sys

from easydict import EasyDict

from basicts.data import UEADataset
from basicts.metrics import accuracy
from basicts.runners import SimpleTimeSeriesClassificationRunner
from basicts.utils import load_dataset_desc

from .arch import iTransformer

sys.path.append(os.path.abspath(__file__ + "/../../.."))

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = "JapaneseVowels"  # Dataset name
desc = load_dataset_desc(os.path.join("UEA", DATA_NAME))
INPUT_LEN = desc["seq_len"]
NUM_CLASSES = desc["num_classes"]
NULL_VAL = 0.0
# Model architecture and parameters
MODEL_ARCH = iTransformer
NUM_NODES = desc["num_nodes"]
MODEL_PARAM = {
    "task_name": "classification",
    "num_classes": NUM_CLASSES,
    "enc_in": NUM_NODES,                        # num nodes
    "dec_in": NUM_NODES,
    "c_out": NUM_NODES,
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN/2,       # start token length used in decoder
    "pred_len": 0,         # prediction sequence length
    "factor": 3, # attn factor
    "d_model": 128,
    "moving_avg": 25,                           # window size of moving average. This is a CRUCIAL hyper-parameter.
    "n_heads": 8,
    "e_layers": 3,                              # num of encoder layers
    "d_ff": 256,
    "distil": True,
    "sigma" : 0.2,
    "dropout": 0.1,
    "freq": "h",
    "use_norm": True,
    "output_attention": False,
    "embed": "timeF",                           # [timeF, fixed, learned]
    "activation": "gelu",
    }
NUM_EPOCHS = 20

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = "An Example Config"
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesClassificationRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = UEADataset
CFG.DATASET.NUM_CLASSES = NUM_CLASSES
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": None, # None for UEA datasets
    # "mode" is automatically set by the runner
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                "Accuracy": accuracy
                            })
CFG.METRICS.TARGET = "Accuracy"
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# CFG.TRAIN.LOSS = nn.CrossEntropyLoss()
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 25, 50],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
