# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DLinear import DLinear, DLinearConfig


def test_dlinear_smoke_test():
    output_len = 64
    input_len = 64
    dlinear_config = DLinearConfig(
        input_len=input_len,
        output_len=output_len,
        individual=False,
    )

    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=DLinear,
            dataset_name="ETTh1_mini",
            model_config=dlinear_config,
            gpus=None,
            num_epochs=5,
            input_len=input_len,
            output_len=output_len,
        )
    )

if __name__ == "__main__":
    test_dlinear_smoke_test()
