from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TimerConfig(BasicTSModelConfig):

    """
    Config class for Timer model.
    """

    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    patch_len: int = field(default=96, metadata={"help": "Patch length."})
    hidden_size: int = field(default=1024, metadata={"help": "Hidden size."})
    n_heads: int = field(default=8, metadata={"help": "Number of query heads in MultiHeadAttention."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=6, metadata={"help": "Number of decoder layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=False, metadata={"help": "Whether to use RevIN."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})
