from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class UMixerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 64
    num_layers: int = 1
    patch_len: int = 24
    stride: int = 24
    dropout: float = 0.1
    use_revin: bool = True
