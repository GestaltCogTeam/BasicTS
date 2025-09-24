from typing import Sequence, List

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.STID import STID, STIDConfig
from basicts.runners.callback import BasicTSCallback, EarlyStopping


def STID_ETTh1(
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
                assert training_config.model.__class__.__name__ == "STID" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be STID and dataset should be ETTh1."
            else:
                model = STID(
                    STIDConfig(
                        input_len=input_len,
                        output_len=output_len,
                        num_features=7,
                        input_hidden_size=2048,
                        if_spatial=False,
                        if_time_in_day=False,
                        if_day_in_week=False
                    )
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus,
                    callbacks=callbacks,
                )
            BasicTSLauncher.launch_training(training_config)

if __name__ == "__main__":
    STID_ETTh1(
        gpus="0",
        callbacks=[EarlyStopping(patience=5)],
    )
