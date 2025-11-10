from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class NonstationaryTransformerConfig(BasicTSModelConfig):

    """
    Config class for NonstationaryTransformer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    label_len: int = field(
        default=None, metadata={"help": "Label length for forecasting task. Only used in forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    proj_hidden_size: int = field(default=256, metadata={"help": "Hidden size of projector layers."})
    n_heads: int = field(default=8, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_encoder_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})
    num_decoder_layers: int = field(
        default=1, metadata={"help": "Number of decoder layers. Only used in forecasting task."})
    num_proj_layers: int = field(default=1, metadata={"help": "Number of projector layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    threshold: float = field(default=80.0, metadata={"help": "Threshold in clamp function to avoid large values."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})
    use_timestamps: bool = field(default=False, metadata={"help": "Whether to use timestamps as tokens."})
    timestamp_sizes: Sequence[int] = field(default=None, metadata={"help": "Sizes of timestamps."})
