from dataclasses import dataclass, field
from typing import List

from basicts.configs import BasicTSModelConfig


@dataclass
class PyraformerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 512
    intermediate_size: int = 512
    key_size: int = 128
    value_size: int = 128
    bottleneck_size: int = 128
    n_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.05
    decoder: str = "FC"
    window_size: List[int] = field(default_factory=lambda: [2, 2, 2])
    inner_size: int = 5
    pyramid_constructor: str = "Bottleneck_Construct"
    truncate: bool = False
    use_tvm: bool = False
    embed: str = "DataEmbedding"
    num_timestamps: int = 0
