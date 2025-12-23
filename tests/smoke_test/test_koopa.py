# pylint: disable=wrong-import-position
import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.Koopa import Koopa, KoopaConfig
from basicts.models.Koopa.callback.koopa_mask_init import KoopaMaskInitCallback


def test_koopa_smoke_test():
    output_len = 48
    input_len = 96
    koopa_config = KoopaConfig(
        input_len=input_len,
        output_len=output_len,
        enc_in=7,
        seg_len=48,
        dynamic_dim=64,
        hidden_dim=512,
        num_blocks=4
    )
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=Koopa,
            dataset_name="ETTh2_mini",
            model_config=koopa_config,
            gpus=None,
            num_epochs=1,
            input_len=input_len,
            output_len=output_len,
            lr=0.001,
            callbacks=[KoopaMaskInitCallback(alpha=0.2)],
        )
    )

if __name__ == "__main__":
    test_koopa_smoke_test()
