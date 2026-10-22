from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class BiMambaConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 128
    intermediate_size: int = 256
    state_size: int = 32
    num_layers: int = 3
    patch_len: int = 24
    stride: int = 12
    conv_size: int = 2
    expand_factor: int = 1
    dropout: float = 0.1
    hidden_act: str = "gelu"
    use_revin: bool = True
    bidirectional: bool = True
    residual: bool = True
    channel_independence: bool = True
    use_sra: bool = False
    correlation: Optional[object] = None
    correlation_threshold: float = 0.6
    padding_patch: str = "end"
    embed_type: int = 0
    position_embedding_type: str = "sincos"
    position_learnable: bool = False
    num_timestamps: int = 0
