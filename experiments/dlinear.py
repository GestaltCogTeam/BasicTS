# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basicts.configs import BasicTSForecastingConfig
from basicts.data import BuiltinTSForecastingDataset
from basicts.launcher import BasicTSLauncher
from basicts.models import DLinear

for output_len in [336]:
    input_len = 336
    model = DLinear({
    "seq_len": input_len,
    "pred_len": output_len,
    "individual": False,
    "enc_in": 7
    })

    dataset = BuiltinTSForecastingDataset("ETTh1", input_len, output_len)

    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=model,
            dataset=dataset,
            gpus="0"
        )
    )


