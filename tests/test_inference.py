import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from argparse import ArgumentParser

from easytorch import launch_runner, Runner


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', default="examples/STID/STID_METR-LA.py", help='training config')
    parser.add_argument('--ckpt', default="checkpoints/STID_200/c3aeadfd9ebb252bc7170a47950b81b8/STID_005.pt", help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()


def main(cfg: dict, runner: Runner, ckpt: str = None):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    runner.load_model(ckpt_path=ckpt)

    runner.test_process(cfg)


if __name__ == '__main__':
    args = parse_args()
    try:
        launch_runner(args.cfg, main, (args.ckpt,), devices=args.gpus)
    except TypeError as e:
        if "launch_training() got an unexpected keyword argument" in repr(e):
            # NOTE: for earlier easytorch version
            launch_runner(args.cfg, main, (args.ckpt,), gpus=args.gpus)
        else:
            raise e
