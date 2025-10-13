# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DCRNN import DCRNN, DCRNNConfig
from basicts.runners.callback import ClipGrad

from basicts.utils import load_meta_description, load_adj

def launch_training(
        training_config: BasicTSForecastingConfig | None = None,
        ):
    BasicTSLauncher.launch_training(training_config)


if __name__ == "__main__":
    dataset_name = 'PEMS08'
    meta_desc = load_meta_description(dataset_name)
    adj_mx, _ = load_adj(dataset_name, "doubletransition")

    regular_settings = meta_desc['regular_settings']
    model_config = DCRNNConfig(
        num_nodes=meta_desc['num_vars'],
        adj_mx=[torch.tensor(adj) for adj in adj_mx],
        seq_len=12,
        horizon=12,
        input_dim=2
    )
    model = DCRNN(model_config)
    training_config = BasicTSForecastingConfig(
        model=model,
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        seed=42,
        gpus='2',
        optimizer=torch.optim.Adam,
        optimizer_params={
            'lr': 0.003,
            'eps': 1e-3
        },
        rescale=regular_settings['rescale'],
        metrics=regular_settings['metrics'],
        null_val=regular_settings['null_val'],
        norm_each_channel=regular_settings['norm_each_channel'],
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            'milestones': [20, 50, 70],
            'gamma': 0.3
        },
        callbacks=[ClipGrad(max_norm=5)],
        batch_size=64
    )
    launch_training(training_config)
