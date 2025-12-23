# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.SegRNN import SegRNN, SegRNNConfig


def test_segrnn_smoke_test():
    output_len = 24
    input_len = 96
    segrnn_config = SegRNNConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=SegRNN,
            dataset_name="ETTh1_mini",
            model_config=segrnn_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001
        )
    )

if __name__ == "__main__":
    test_segrnn_smoke_test()
