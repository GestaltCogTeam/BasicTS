# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import basicts


def parse_args():
    parser = ArgumentParser(description='Inference a time series forecasting model in BasicTS framework!')
    # enter your config file path
    parser.add_argument('-cfg', '--config', default='baselines/STID/PEMS04.py', help='training config')
    # enter your own checkpoint file path
    parser.add_argument('-ckpt', '--checkpoint', default='checkpoints/STID/PEMS04_100_12_12/5684a53d44870276f5fb6522f26cf035/STID_best_val_MAE.pt')
    parser.add_argument('-i', '--input_data_file_path', default='./in.csv')
    parser.add_argument('-o', '--output_data_file_path', default='./out.csv')
    parser.add_argument('-g', '--gpus', default='0')
    parser.add_argument('-d', '--device_type', default='cpu')
    parser.add_argument('-ctx', '--context_length', type=int, default=72, help='context length for inference, only used for utsf models')
    parser.add_argument('-pred', '--prediction_length', type=int, default=24, help='prediction length for inference, only used for utsf models')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    basicts.launch_inference(cfg=args.config, ckpt_path=args.checkpoint, input_data_file_path=args.input_data_file_path,
                             output_data_file_path=args.output_data_file_path, device_type=args.device_type, gpus=args.gpus,
                             context_length=args.context_length, prediction_length=args.prediction_length)
