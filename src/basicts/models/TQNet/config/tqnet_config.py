from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TQNetConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    cycle_len: int = 24
    hidden_size: int = 512
    dropout: float = 0.0
    use_revin: bool = True
    n_heads: int = 4
