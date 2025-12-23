from basicts import BasicTSLauncher
from basicts.configs import BasicTSClassificationConfig
from basicts.models.iTransformer import (iTransformerConfig,
                                         iTransformerForClassification)


def main():

    model_config = iTransformerConfig(
        input_len=144,
        num_features=9,
        num_classes=25
        )

    BasicTSLauncher.launch_training(BasicTSClassificationConfig(
        model=iTransformerForClassification,
        model_config=model_config,
        dataset_name="ArticularyWordRecognition",
        gpus="0"
    ))


if __name__ == "__main__":
    main()
