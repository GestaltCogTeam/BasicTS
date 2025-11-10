from dataclasses import dataclass, field
from typing import Literal, Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class TimeMixerConfig(BasicTSModelConfig):

    """
    Config class for TimeMixer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    down_sampling_window: int = field(default=2, metadata={"help": "Down sampling window."})
    down_sampling_layers: int = field(default=3, metadata={"help": "Down sampling layers."})
    down_sampling_method: Literal["avg", "max", "conv"] = field(
        default="avg", metadata={"help": "Down sampling method. 'avg', 'max' or 'conv'."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    channel_independence: bool = field(default=True, metadata={"help": "Whether to use channel independence."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    decomp_method: Literal["moving_avg", "dft_decomp"] = field(
        default="moving_avg", metadata={"help": "Decomposition method. 'moving_avg' or 'dft_decomp'."})
    moving_avg: int = field(default=25, metadata={"help": "Kernel size of moving average."})
    top_k: int = field(default=5, metadata={"help": "Top-k of amplitude when use DFTDecomposition."})
    use_timestamps: bool = field(default=False, metadata={"help": "Whether to use timestamps as tokens."})
    timestamp_sizes: Sequence[int] = field(default=None, metadata={"help": "Sizes of timestamps."})
