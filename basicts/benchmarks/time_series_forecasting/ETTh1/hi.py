from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.HI import HI, HIConfig
from basicts.runners.callback import BasicTSCallback, NoBP


def HI_ETTh1(
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
                assert training_config.model.__class__.__name__ == "HI" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be HI and dataset should be ETTh1."
            else:
                model = HI(
                    HIConfig(
                        input_len=input_len,
                        output_len=output_len
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
                    if not any(isinstance(item, NoBP) for item in callbacks):
                        callbacks.append(NoBP())
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
