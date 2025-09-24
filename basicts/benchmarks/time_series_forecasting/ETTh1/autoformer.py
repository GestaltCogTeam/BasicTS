from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.Autoformer import Autoformer, AutoformerConfig
from basicts.runners.callback import BasicTSCallback


def Autoformer_ETTh1(
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
                assert training_config.model.__class__.__name__ == "Autoformer" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be Autoformer and dataset should be ETTh1."
            else:
                model = Autoformer(
                    AutoformerConfig(
                        input_len,
                        output_len,
                        output_len // 2,
                        7,
                    )
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
