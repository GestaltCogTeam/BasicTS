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
from basicts.models.D2STGNN import D2STGNN, D2STGNNConfig
from basicts.runners.callback import ClipGrad, CurriculumLearning

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
    model_config = D2STGNNConfig(
        num_nodes=meta_desc['num_vars'],
        adjs=[torch.tensor(adj) for adj in adj_mx]
    )
    model = D2STGNN(model_config)
    training_config = BasicTSForecastingConfig(
        model=model,
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        seed=42,
        gpus='2',
        optimizer=torch.optim.Adam,
        optimizer_params={
            'lr': 0.002,
            'weight_decay': 1e-5,
            'eps': 1e-8
        },
        rescale=regular_settings['rescale'],
        metrics=regular_settings['metrics'],
        null_val=regular_settings['null_val'],
        norm_each_channel=regular_settings['norm_each_channel'],
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            'milestones': [1, 30, 38, 46, 54, 62, 70, 80],
            'gamma': 0.5
        },
        callbacks=[ClipGrad(max_norm=5),
                   CurriculumLearning(prediction_length=12, warm_up_epochs=30, cl_epochs=3)],
        batch_size=16
    )
    launch_training(training_config)
