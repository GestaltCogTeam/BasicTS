from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.TimeMixer import TimeMixerForForecasting, TimeMixerConfig
from basicts.runners.callback import BasicTSCallback
from basicts.metrics import masked_mse


def TimeMixer_ETTh1(
        input_lens: Sequence[int] = (336,),
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
                model = TimeMixerForForecasting(
                    TimeMixerConfig(
                        input_len=input_len,
                        output_len=output_len,
                        num_features=7,
                        hidden_size=16,
                        intermediate_size=32,
                        down_sampling_layers=3,
                        down_sampling_window=2,
                        num_layers=2,
                    )
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    metrics=["MSE", "MAE"],
                    loss=masked_mse,
                    seed=seed,
                    gpus=gpus,
                    optimizer_params={"lr": 0.01}
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
