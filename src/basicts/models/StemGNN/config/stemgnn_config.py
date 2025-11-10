from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class StemGNNConfig(BasicTSModelConfig):
    """
    Config class for StemGNN model.
    """
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_blocks: int = field(default=2, metadata={"help": "Number of StemGNN Block."})
    hidden_size: int = field(default=5, metadata={"help": "Hyperparameter of STemGNN which controls the parameter number of hidden layers."})
    dropout: float = field(default=0.5, metadata={"help": "Dropout rate."})
    leaky_rate: float = field(default=0.2, metadata={"help": "LeakyReLU activation function parameters."})
