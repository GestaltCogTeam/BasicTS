# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basicts.configs import BasicTSFoundationModelConfig
from basicts.data import BLASTDatasetWoMixUp
from basicts.launcher import BasicTSLauncher
from basicts.models import ChronosBolt
from basicts.runners.taskflow.forecasting_taskflow import BasicTSForecastingTaskFlow
from basicts.runners.callback import GradAccumulation, ClipGrad


model = ChronosBolt(
    model_id="./baselines/ChronosBolt/ckpt/chronos-bolt-small",
    from_pretrained=False,
    device_map="cpu",
)

dataset = BLASTDatasetWoMixUp(context_length=1024, target_length=64, num_valid_samples=1000)
config = BasicTSFoundationModelConfig(
            taskflow=BasicTSForecastingTaskFlow(),
            model=model,
            dataset=dataset,
            gpus="0",
            callbacks=[GradAccumulation(10), ClipGrad(1.0)]
        )

BasicTSLauncher.launch_training(config)


