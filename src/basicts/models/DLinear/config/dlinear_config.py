from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DLinearConfig(BasicTSModelConfig):

    """
    Config class for DLinear model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=1, metadata={"help": "Number of features."})
    moving_avg: int = field(default=25, metadata={"help": "Kernel size of moving average decomposition."})
    stride: int = field(default=1, metadata={"help": "Stride of moving average decomposition."})
    individual: bool = field(default=False, metadata={"help": "If use individual linear layer for each channel."})
