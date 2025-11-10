from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class CrossformerConfig(BasicTSModelConfig):

    """
    Config class for Crossformer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_layers: int = field(default=2, metadata={"help": "Number of layers (scales)."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    win_size: int = field(default=2, metadata={"help": "Window size."})
    factor: int = field(default=10, metadata={"help": "Factor of the router in TwoStageAttention."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    n_heads: int = field(default=8, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    dropout: float = field(default=0.05, metadata={"help": "Dropout rate."})
    baseline: bool = field(default=False, metadata={"help": "Whether to add an averaged input to prediction."})
