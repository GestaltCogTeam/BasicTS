# Run a baseline model in BasicTS framework.
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/STID/PEMS04.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == "__main__":
    main()
