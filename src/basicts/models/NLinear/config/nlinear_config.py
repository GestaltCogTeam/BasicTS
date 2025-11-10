from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class NLinearConfig(BasicTSModelConfig):

    """
    Config class for NLinear model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
