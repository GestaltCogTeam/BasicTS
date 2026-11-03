from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DSFormerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    num_layers: int = 1
    dropout: float = 0.05
    n_heads: int = 4
    num_samp: int = 3
    use_node_embedding: bool = True
    use_revin: bool = True
