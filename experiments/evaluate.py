import os
import sys
import time
sys.path.append(os.path.abspath(__file__ + '/../..'))
from argparse import ArgumentParser

import basicts

def parse_args():
    parser = ArgumentParser(description="Evaluate time series forecasting model in BasicTS framework!")
    parser.add_argument("-cfg", "--config", default="examples/complete_config.py", help="training config")
    parser.add_argument("-ckpt", "--checkpoint", default="")
    parser.add_argument("-g", "--gpus", default="0")
    parser.add_argument("-d", "--device_type", default="gpu")
    parser.add_argument("-b", "--batch_size", default=None)
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    basicts.launch_evaluation(cfg=args.config, ckpt_path=args.checkpoint, device_type=args.device_type, gpus=args.gpus, batch_size=args.batch_size)
