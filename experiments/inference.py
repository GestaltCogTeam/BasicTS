import os
import sys
import time
sys.path.append(os.path.abspath(__file__ + '/../..'))
from argparse import ArgumentParser

from basicts import launch_runner, BaseRunner


def inference(cfg: dict, runner: BaseRunner, ckpt: str = None, batch_size: int = 1):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    # init model
    cfg.TEST.DATA.BATCH_SIZE = batch_size
    runner.model.eval()
    runner.setup_graph(cfg=cfg, train=False)
    # load model checkpoint
    runner.load_model(ckpt_path=ckpt)
    # inference & speed
    t0 = time.perf_counter()
    runner.test_pipline(cfg)
    elapsed = time.perf_counter() - t0

    print('##############################')
    runner.logger.info('%s: %0.8fs' % ('Speed', elapsed))
    runner.logger.info('# Param: {0}'.format(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)))

if __name__ == '__main__':
    MODEL_NAME = 'AGCRN'
    DATASET_NAME = 'PEMS03'
    BATCH_SIZE = 32
    GPUS = '2'

    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-m', '--model', default=MODEL_NAME, help='model name')
    parser.add_argument('-d', '--dataset', default=DATASET_NAME, help='dataset name')
    parser.add_argument('-g', '--gpus', default=GPUS, help='visible gpus')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    args = parser.parse_args()

    cfg_path = 'baselines/{0}/{1}.py'.format(args.model, args.dataset)
    ckpt_path = 'ckpt/{0}/{1}/{0}_best_val_MAE.pt'.format(args.model, args.dataset)

    launch_runner(cfg_path, inference, (ckpt_path, args.batch_size), devices=args.gpus)
