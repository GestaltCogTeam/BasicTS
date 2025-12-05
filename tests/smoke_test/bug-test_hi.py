# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.HI.arch.hi_arch import HI
from basicts.models.HI.config.hi_config import HIConfig
from basicts.runners.callback import NoBP


def test_hi_smoke_test():
    output_len = 24
    input_len = 96
    hi_config = HIConfig(
        input_len=input_len,
        output_len=output_len,
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=HI,
            dataset_name="ETTh1_mini",
            model_config=hi_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001,
            callbacks=[NoBP()],
        )
    )

if __name__ == "__main__":
    test_hi_smoke_test()
