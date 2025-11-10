from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class TimesNetConfig(BasicTSModelConfig):

    """
    Config class for TimesNet model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    num_kernels: int = field(default=3, metadata={"help": "Number of kernels in Inception block."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    top_k: int = field(default=5, metadata={"help": "Top-k of amplitude in FFT."})
    use_timestamps: bool = field(default=False, metadata={"help": "Whether to use timestamps as tokens."})
    timestamp_sizes: Sequence[int] = field(default=None, metadata={"help": "Sizes of timestamps."})
