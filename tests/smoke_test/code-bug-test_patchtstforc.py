# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts import BasicTSLauncher
from basicts.configs import BasicTSClassificationConfig
from basicts.models.PatchTST import PatchTSTForClassification, PatchTSTConfig


def test_patchtstforc_smoke_test():

    model_config = PatchTSTConfig(
        input_len=144,
        num_features=9,
        num_classes=25
        )

    BasicTSLauncher.launch_training(BasicTSClassificationConfig(
        model=PatchTSTForClassification,
        model_config=model_config,
        dataset_name="ArticularyWordRecognition_mini",
        gpus=None,
        batch_size=16,
        num_epochs=1,
    ))


if __name__ == "__main__":
    test_patchtstforc_smoke_test()
