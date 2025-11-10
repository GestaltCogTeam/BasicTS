from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DUETConfig(BasicTSModelConfig):

    """
    Config class for DUET model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    channel_independence: bool = field(default=True, metadata={"help": "Whether to use channel independence."})

    # MoE architecture
    num_experts: int = field(default=4, metadata={"help": "Number of experts."})
    noisy_gating: bool = field(default=True, metadata={"help": "Whether to use noisy gating."})
    moving_avg: int = field(default=25, metadata={"help": "Kernel size of moving average in experts."})
    top_k: int = field(default=1, metadata={"help": "Top-k of experts."})
    loss_coef: float = field(default=1.0, metadata={"help": "Coefficient of MoE load balance loss."})

    # Transformer encoder
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    n_heads: int = field(default=8, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=2048, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})
    dropout: float = field(default=0.2, metadata={"help": "Dropout rate."})

    # Forecasting head
    fc_dropout: float = field(default=0.2, metadata={"help": "Dropout rate of FC layer."})
