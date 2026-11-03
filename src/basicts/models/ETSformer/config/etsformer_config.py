from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class ETSformerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 256
    intermediate_size: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    top_k: int = 3
    sigma: float = 0.2
    dropout: float = 0.05
    hidden_act: str = "sigmoid"
    output_attentions: bool = False
    num_timestamps: int = 0
