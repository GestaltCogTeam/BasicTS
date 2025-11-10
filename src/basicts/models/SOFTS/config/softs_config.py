from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SOFTSConfig(BasicTSModelConfig):

    """
    Config class for SOFTS model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    core_size: int = field(default=128, metadata={"help": "Core size."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Hidden activation function."})
    num_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.05, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
