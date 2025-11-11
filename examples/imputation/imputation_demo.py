from basicts import BasicTSLauncher
from basicts.configs import BasicTSImputationConfig
from basicts.models.iTransformer import (iTransformerConfig,
                                         iTransformerForReconstruction)


def main():

    model_config = iTransformerConfig(num_features=7)

    BasicTSLauncher.launch_training(BasicTSImputationConfig(
        model=iTransformerForReconstruction,
        model_config=model_config,
        dataset_name="ETTh1",
        input_len=336,
        mask_ratio=0.25,
        gpus="0"
    ))


if __name__ == "__main__":
    main()
