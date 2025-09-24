from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.TimesNet import TimesNetForForecasting, TimesNetConfig
from basicts.runners.callback import BasicTSCallback
from basicts.metrics import masked_mse


def TimesNet_ETTh1(
        input_lens: Sequence[int] = (96,),
        output_lens: Sequence[int] = (96, 192, 336, 720),
        training_config: BasicTSForecastingConfig | None = None,
        gpus: str | None = None,
        callbacks: List[BasicTSCallback] = None,
        seed: int = 42,
        ):

    for input_len in input_lens:
        for output_len in output_lens:
            if training_config is not None:
                assert training_config.model.__class__.__name__ == "iTransformer" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be iTransformer and dataset should be ETTh1."
            else:
                model = TimesNetForForecasting(
                    TimesNetConfig(
                        input_len=input_len,
                        output_len=output_len,
                        num_features=7,
                        top_k=5,
                        hidden_size=16,
                        intermediate_size=32,
                        num_kernels=6,
                        num_layers=2,
                        dropout=0.05,
                        use_timestamps=True,
                        timestamp_sizes=(24, 7, 31, 366),
                    )
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus,
                    loss=masked_mse,
                    metrics=["MSE", "MAE"]
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
