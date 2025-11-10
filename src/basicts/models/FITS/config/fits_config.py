from dataclasses import dataclass, field
from typing import Callable, Literal, Union

from basicts.configs import BasicTSModelConfig


@dataclass
class FITSConfig(BasicTSModelConfig):

    """
    Config class for FITS model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=1, metadata={"help": "Number of features."})
    individual: bool = field(default=False, metadata={"help": "If use individual linear layer for each channel."})
    cut_freq: int = field(default=None, metadata={"help": "Cut-off frequency."})
    base_period: int = field(default=24, metadata={"help": "Base period."})
    h_order: int = field(default=6, metadata={"help": "Order of harmonics."})
    loss: Union[str, Callable] = field(default="MSE", metadata={"help": "Loss function."})
    training_mode: Literal["y", "xy"] = field(default="y",
        metadata={"help": "Training mode. `y` denotes computing loss over target timesteps." \
                  "`xy` denotes computing loss over inputs and target timesteps."})
