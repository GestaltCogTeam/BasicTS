# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.Autoformer.arch.autoformer_arch import Autoformer
from basicts.models.Autoformer.config.autoformer_config import AutoformerConfig


def test_autoformer_smoke_test():
    output_len = 24
    input_len = 96
    autoformer_config = AutoformerConfig(
        input_len=input_len,
        output_len=output_len,
        label_len=input_len/2,
        num_features=7,
        use_timestamps=True,
        timestamp_sizes=[24, 7, 31, 366],
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=Autoformer,
            dataset_name="ETTh1",
            model_config=autoformer_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001,
            use_timestamps=True,
        )
    )

if __name__ == "__main__":
    test_autoformer_smoke_test()
