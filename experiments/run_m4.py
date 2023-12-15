# 1. 给定方法，例如MLP； 给定是否保存结果，给定是否保留预测结果；
# 2. 检查是否存在配置文件生成器，例如baselines/MLP/M4_base.py
# 3. 循环获取CFG，进行训练并保存结果
# 4. M4 Summary
# 5. 保存结果、删除预测结果

import os
import sys
import importlib
from argparse import ArgumentParser
# TODO: remove it when basicts can be installed by pip
project_dir = os.path.abspath(__file__ + "/../..")
sys.path.append(project_dir)
import torch
from basicts import launch_training
from basicts.utils import m4_summary
from easytorch.utils.logging import logger_initialized
from basicts.utils.logging import clear_loggers

torch.set_num_threads(3) # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--config", default="baselines/STID_M4/M4.py", help="training config template")
    parser.add_argument("-c", "--config", default="baselines/MLP/M4.py", help="training config template")
    parser.add_argument("-g", "--gpus", default="3", help="visible gpus")
    parser.add_argument("--save_evaluation", default=True, help="if save evaluation results")
    parser.add_argument("--save_prediction", default=False, help="if save prediction results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg_generator_file = args.config[:-3].replace("/", ".")

    # training
    get_cfg = importlib.import_module(cfg_generator_file, package=project_dir).get_cfg
    seasonal_patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    for seasonal_pattern in seasonal_patterns:
        cfg = get_cfg(seasonal_pattern)
        launch_training(cfg, args.gpus)
        clear_loggers()

    # evaluating
    save_dir = os.path.abspath(args.config + "/..")
    result = m4_summary(save_dir, project_dir) # pd.DataFrame

    # save results
    if not args.save_prediction: os.system("rm -rf {0}/M4_*.npy".format(save_dir))
    if args.save_evaluation: result.to_csv("{0}/M4_summary.csv".format(save_dir), index=False)
    else: os.system("rm {0}/M4_summary.csv".format(save_dir))
    
    # print results
    print(result)
