# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.Informer import Informer, InformerConfig


def test_informer_smoke_test():

    output_len = 48
    input_len = 96
    informer_config = InformerConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        label_len=input_len / 2,
        use_timestamps=True,
        timestamp_sizes=[24, 7, 31, 366],

    )

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
            model=Informer,
            dataset_name="ETTh1_mini",
            model_config=informer_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001
    ))


if __name__ == "__main__":
    test_informer_smoke_test()
