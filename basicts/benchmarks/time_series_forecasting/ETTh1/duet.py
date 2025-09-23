from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.DUET import DUET, DUETConfig, DUETComputeLoss
from basicts.runners.callback import BasicTSCallback, EarlyStopping


def DUET_ETTh1(
        input_lens: Sequence[int] = (512,),
        output_lens: Sequence[int] = (96, 192, 336, 720),
        training_config: BasicTSForecastingConfig | None = None,
        gpus: str | None = None,
        callbacks: List[BasicTSCallback] = None,
        seed: int = 42,
        ):

    for input_len in input_lens:
        for output_len in output_lens:
            if training_config is not None:
                assert training_config.model.__class__.__name__ == "DUET" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be DUET and dataset should be ETTh1."
            else:
                model = DUET(
                    DUETConfig(
                        input_len=input_len,
                        output_len=output_len,
                        num_features=7,
                        hidden_size=512,
                        intermediate_size=512,
                        num_experts=2,
                        n_heads=1,
                        top_k=1,
                        num_layers=1,
                        dropout=0.5
                    )
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus,
                    batch_size=32,
                    optimizer_params={"lr": 0.0005},
                    metrics=["MSE", "MAE"],
                    callbacks=[DUETComputeLoss(), EarlyStopping(5)]
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
