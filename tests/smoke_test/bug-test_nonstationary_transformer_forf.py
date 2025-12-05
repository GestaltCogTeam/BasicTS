# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.NonstationaryTransformer import NonstationaryTransformerForForecasting, NonstationaryTransformerConfig


def test_nonstationary_transformer_for_forecasting_smoke_test():
    output_len = 24
    input_len = 96
    nonstationary_transformer_config = NonstationaryTransformerConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        label_len=input_len // 2,
        use_timestamps=True,
        timestamp_sizes=[24, 7, 31, 366],
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=NonstationaryTransformerForForecasting,
            dataset_name="ETTh1_mini",
            model_config=nonstationary_transformer_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001
        )
    )

if __name__ == "__main__":
    test_nonstationary_transformer_for_forecasting_smoke_test()
