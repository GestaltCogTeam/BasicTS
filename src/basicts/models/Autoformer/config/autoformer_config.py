from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class AutoformerConfig(BasicTSModelConfig):

    """
    Config class for Autoformer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    label_len: int = field(default=None, metadata={"help": "Label length for decoder."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    n_heads: int = field(default=8, metadata={"help": "Number of heads in multi-head attention."})
    factor: int = field(default=3, metadata={"help": "Factor in auto-correlation."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_encoder_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})
    num_decoder_layers: int = field(default=1, metadata={"help": "Number of decoder layers."})
    moving_avg: int = field(default=25, metadata={"help": "Kernel size of moving average decomposition."})
    dropout: float = field(default=0.05, metadata={"help": "Dropout rate."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})
    use_timestamps: bool = field(default=False, metadata={"help": "Whether to use timestamps as tokens."})
    timestamp_sizes: Sequence[int] = field(default=None, metadata={"help": "Sizes of timestamps."})
