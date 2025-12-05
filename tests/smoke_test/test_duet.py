# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DUET import DUET, DUETConfig
from basicts.runners.callback import AddAuxiliaryLoss


def test_duet_smoke_test():
    output_len = 24
    input_len = 96
    duet_config = DUETConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=DUET,
            dataset_name="ETTh1_mini",
            model_config=duet_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001,
            callbacks=[AddAuxiliaryLoss()]
        )
    )

if __name__ == "__main__":
    test_duet_smoke_test()
