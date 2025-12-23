from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import SelectiveLearning


if __name__ == "__main__":

    cb = SelectiveLearning(
                r_u=0.3,
                r_a=0.3,
                estimator=DLinear,
                estimator_config=DLinearConfig(input_len=336, output_len=336),
                ckpt_path="checkpoints/DLinear/ETTh1_100_336_336/1f037d3a0fb4a6de40ce3dcb2656b136/DLinear_best_val_MSE.pt"
            )
    
    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=iTransformerForForecasting,
            input_len=336,
            output_len=336,
            use_timestamps=False,
            model_config=iTransformerConfig(
                input_len=336,
                output_len=336,
                num_features=7),
            dataset_name="ETTh1",
            gpus="0",
            callbacks=[cb],
    ))