from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mse
from basicts.models.iTransformer import (iTransformerConfig,
                                         iTransformerForForecasting)
from basicts.runners.callback import EarlyStopping, GradientClipping


def main():

    # train iTransformer on ETTh1
    # run 4 experiments with different `input_len` and `output_len`
    for input_len in [96, 192, 336, 720]:
        for output_len in [96, 192, 336, 720]:

            # config iTransformer
            model_config = iTransformerConfig(
                num_features=7,
                hidden_size=32,
                intermediate_size=32,
                n_heads=1,
                num_layers=1,
                dropout=0.1,
                use_revin=True
            )

            BasicTSLauncher.launch_training(BasicTSForecastingConfig(
                model=iTransformerForForecasting,
                model_config=model_config,
                dataset_name="ETTh1",
                input_len=input_len,
                output_len=output_len,
                gpus="0",
                callbacks=[EarlyStopping(), GradientClipping(1.0)], # use callbacks
                seed=233,
                num_epochs=100,
                batch_size=64,
                metrics=["MSE", "MAE"],
                loss=masked_mse,
                optimizer_params={
                    "lr": 5e-4
                },
                lr_scheduler=MultiStepLR,
                lr_scheduler_params={
                    "milestones": [25, 50],
                    "gamma": 0.5
                }
            ))


if __name__ == "__main__":
    main()
