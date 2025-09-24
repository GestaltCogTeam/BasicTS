from typing import List, Sequence

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.Crossformer import Crossformer
from basicts.runners.callback import BasicTSCallback


def Crossformer_ETTh1(
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
                assert training_config.model.__class__.__name__ == "DLinear" \
                    and training_config.dataset_name == "ETTh1", \
                    "The model should be DLinear and dataset should be ETTh1."
            else:
                model = Crossformer(
                    data_dim=7,
                    in_len=input_len,
                    out_len=output_len,
                    seg_len=24,
                    win_size=2,
                    factor=10,
                    hidden_size=256,
                    d_ff = 256,
                    n_heads=1,
                    e_layers=2,
                    dropout=0.2,
                    baseline = False,
                )
                training_config = BasicTSForecastingConfig(
                    model=model,
                    dataset_name="ETTh1",
                    input_len=input_len,
                    output_len=output_len,
                    seed=seed,
                    gpus=gpus,
                    batch_size=64,
                    optimizer_params={"lr": 0.0005},
                    metrics=["MSE", "MAE"],
                )
                if callbacks is not None:
                    training_config.callbacks = callbacks
            BasicTSLauncher.launch_training(training_config)
