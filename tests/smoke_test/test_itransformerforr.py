# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts import BasicTSLauncher
from basicts.configs import BasicTSImputationConfig
from basicts.models.iTransformer import (iTransformerConfig,
                                         iTransformerForReconstruction)


def test_itransformerforr_smoke_test():
    input_len=32
    model_config = iTransformerConfig(
        input_len=input_len,
        num_features=7
        )

    BasicTSLauncher.launch_training(BasicTSImputationConfig(
        model=iTransformerForReconstruction,
        model_config=model_config,
        dataset_name="ETTh1_mini",
        mask_ratio=0.25,
        gpus=None,
        batch_size=16,
        input_len=input_len,
        num_epochs=5,
    ))


if __name__ == "__main__":
    test_itransformerforr_smoke_test()
