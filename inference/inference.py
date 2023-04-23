import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from argparse import ArgumentParser

from basicts import launch_runner, BaseRunner


def inference(cfg: dict, runner: BaseRunner, ckpt: str = None):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    # init model
    runner.model.eval()
    runner.setup_graph(cfg=cfg, train=False)
    # load model checkpoint
    runner.load_model(ckpt_path=ckpt)
    # inference
    runner.test_process(cfg)


if __name__ == '__main__':
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', default="examples/DCRNN/DCRNN_METR-LA.py", help='training config')
    parser.add_argument('--ckpt', default="checkpoints/DCRNN_100/5b638d804a4ad3d2b81de22adcdc2184/DCRNN_best_val_MAE.pt", help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument("--gpus", default="0", help="visible gpus")
    args = parser.parse_args()
    launch_runner(args.cfg, inference, (args.ckpt,), devices=args.gpus)
