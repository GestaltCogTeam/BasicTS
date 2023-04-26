import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(1) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="examples/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/LSCGF/LSCGF_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/DCRNN/DCRNN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/GTS/GTS_PEMS03.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/HI/HI_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_PEMS04_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/Linear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/DLinear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Linear/NLinear_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_ETTm1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_ETTh2.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_Electricity.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_ETTh1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_Weather.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_ExchangeRate.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/MTGNN/MTGNN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/MegaCRN/MegaCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_ETTm1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_ETTm1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_ETTm1.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_METR-LA.py", help="training config")
    parser.add_argument("-c", "--cfg", default="examples/Triformer/Triformer_ETTh1.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_METR-LA_in96_out96.py", help="training config")
    parser.add_argument("--gpus", default="1", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
