# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Sequence
import torch

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.AGCRN import AGCRN, AGCRNConfig

from basicts.runners.callback import BasicTSCallback
from basicts.utils import load_meta_description

def launch_training(
        training_config: BasicTSForecastingConfig | None = None,
        ):
    BasicTSLauncher.launch_training(training_config)


if __name__ == "__main__":
    dataset_name = 'PEMS08'
    meta_desc = load_meta_description(dataset_name)
    regular_settings = meta_desc['regular_settings']
    model_config = AGCRNConfig(
        input_dim=1,
        output_lens=12,
        rnn_units=64,
        output_dim=1,
        num_layers=2,
        embed_dim=10,
        cheb_k=2,
        num_nodes=meta_desc['num_vars'],
    )
    model = AGCRN(model_config)
    training_config = BasicTSForecastingConfig(
        model=model,
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        seed=42,
        gpus='0',
        optimizer=torch.optim.Adam,
        optimizer_params={
            'lr': 0.003 # 需要调整
        },
        rescale=regular_settings['rescale'],
        metrics=regular_settings['metrics'],
        null_val=regular_settings['null_val'],
        norm_each_channel=regular_settings['norm_each_channel'],
        lr_scheduler=None
    )
    launch_training(training_config)
