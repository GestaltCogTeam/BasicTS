# Run a baseline model in BasicTS framework.


import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(3) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default="baselines/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="baselines/STGODE/STGODE_PEMS08.py", help="training config")
    parser.add_argument("-g", "--gpus", default="3", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
