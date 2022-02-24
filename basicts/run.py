import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from argparse import ArgumentParser
from easytorch.easytorch import launch_training

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework based on EasyTorch!')
    parser.add_argument('-opt', '--cfg', required=True, help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/HI/HI_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS04.py', help='training config')
    parser.add_argument('--gpus', default=None, help='visible gpus')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
