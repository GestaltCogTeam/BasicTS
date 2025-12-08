from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TiDEConfig(BasicTSModelConfig):
    """
    Config class for TiDE model.
    """
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: list = field(default=256, metadata={"help": "Hidden size."})
    dropout: float = field(default=0.3, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    intermediate_size: int = field(default=256, metadata={"help": "Intermediate size of FFN layers."})
    num_encoder_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "Number of decoder layers."})
    timestamps_encode_size: int = field(default=2, metadata={"help": "Encoding size of Timestamps."})
    num_timestamps: str = field(default= 4 , metadata={"help": "Sizes of timestamps used."})
    bias: bool = field(default=True, metadata={"help": "Whether to use bias."})
