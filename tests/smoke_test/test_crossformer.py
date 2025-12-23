# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.Crossformer import Crossformer, CrossformerConfig


def test_crossformer_smoke_test():
    output_len = 24
    input_len = 96
    crossformer_config = CrossformerConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        patch_len=24,
        hidden_size=256,
        intermediate_size=512,
        n_heads=4,
        dropout=0.2,
        baseline=False,
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=Crossformer,
            dataset_name="ETTh1_mini",
            model_config=crossformer_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001,
        )
    )

if __name__ == "__main__":
    test_crossformer_smoke_test()
