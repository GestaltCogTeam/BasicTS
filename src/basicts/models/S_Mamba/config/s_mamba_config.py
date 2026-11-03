from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SMambaConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 512
    intermediate_size: int = 512
    state_size: int = 16
    num_layers: int = 3
    conv_size: int = 2
    expand_factor: int = 1
    dropout: float = 0.1
    hidden_act: str = "gelu"
    use_norm: bool = True
    num_timestamps: int = 4
    time_of_day_size: int = 96
