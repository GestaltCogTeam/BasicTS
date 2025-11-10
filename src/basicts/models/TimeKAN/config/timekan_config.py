from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TimeKANConfig(BasicTSModelConfig):

    """
    Config class for TimeKAN model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    begin_order: int = field(default=0, metadata={"help": "Begin order of ChebyKAN."})
    down_sampling_window: int = field(default=2, metadata={"help": "Down sampling window."})
    down_sampling_layers: int = field(default=3, metadata={"help": "Down sampling layers."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
