from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SegRNNConfig(BasicTSModelConfig):

    """
    Config class for SegRNN model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    seg_len: int = field(default=16, metadata={"help": "Segment (patch) length."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of RNN layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for attention layers."})
