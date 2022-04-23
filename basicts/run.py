import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from argparse import ArgumentParser
from easytorch.easytorch import launch_training

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework based on EasyTorch!')
    # parser.add_argument('-opt', '--cfg', required=True, help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/HI/HI_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/HI/HI_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/HI/HI_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/HI/HI_PEMS08.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/Stat/Stat_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/Stat/Stat_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/Stat/Stat_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/Stat/Stat_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/DCRNN/DCRNN_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/DCRNN/DCRNN_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/DCRNN/DCRNN_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/DCRNN/DCRNN_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/BasicMTS/BasicMTS_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/BasicMTS/BasicMTS_PEMS04.py', help='training config')
    parser.add_argument('-opt', '--cfg', default='basicts/options/BasicMTS/BasicMTS_PEMS03.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS03.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS07.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/GraphWaveNet/GraphWaveNet_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/D2STGNN/D2STGNN_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/D2STGNN/D2STGNN_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/D2STGNN/D2STGNN_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/D2STGNN/D2STGNN_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/MTGNN/MTGNN_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/MTGNN/MTGNN_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/MTGNN/MTGNN_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/MTGNN/MTGNN_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/AGCRN/AGCRN_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/AGCRN/AGCRN_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/AGCRN/AGCRN_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/AGCRN/AGCRN_PEMS08.py', help='training config')

    # parser.add_argument('-opt', '--cfg', default='basicts/options/STNorm/STNorm_METR-LA.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/STNorm/STNorm_PEMS-BAY.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/STNorm/STNorm_PEMS03.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/STNorm/STNorm_PEMS04.py', help='training config')
    # parser.add_argument('-opt', '--cfg', default='basicts/options/STNorm/STNorm_PEMS08.py', help='training config')

    parser.add_argument('--gpus', default=None, help='visible gpus')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
