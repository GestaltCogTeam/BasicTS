from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class iTransformerConfig(BasicTSModelConfig):

    """
    Config class for iTransformer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    n_heads: int = field(default=1, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})
