from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.iTransformer import iTransformerForForecasting, iTransformerConfig
from basicts.runners.callback import BasicTSCallback


def iTransformer_ETTh1(
        input_lens: Sequence[int] = (336,),
        output_lens: Sequence[int] = (336,),
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
                training_config = BasicTSForecastingConfig(
                    model=iTransformerForForecasting,
                    model_config=iTransformerConfig(
                        input_len=input_len,
                        output_len=output_len,
                        num_features=7,
                        hidden_size=32,
                        intermediate_size=32
                    ),
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
