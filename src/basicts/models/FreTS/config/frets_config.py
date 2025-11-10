from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class FreTSConfig(BasicTSModelConfig):

    """
    Config class for FreTS model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    embed_size: int = field(default=128, metadata={"help": "Embedding size of the input sequence."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size of the FC layer."})
    hidden_act: str = field(default="leaky_relu", metadata={"help": "Activation function of the FC layer."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate of the FC layer."})
    scale: float = field(default=0.02, metadata={"help": "Scale factor of the frequency MLP layers."})
    sparsity_threshold: float = field(
        default=0.01, metadata={"help": "Sparsity threshold of the frequency MLP layers."})
    channel_independence: bool = field(
        default=False, metadata={"help": "If use channel independence strategy." \
                                         "If False, apply channel-wise frequency MLP layers."})
