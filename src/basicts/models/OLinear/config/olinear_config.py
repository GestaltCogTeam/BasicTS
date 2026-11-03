from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class OLinearConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = 256
    intermediate_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    hidden_act: str = "gelu"
    embed_size: int = 8
    patch_len: int = 16
    stride: int = 8
    channel_independent_basis: bool = False
    cka: bool = False
    use_revin: bool = True
    input_basis_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a NumPy input basis fitted on training data. None selects the DCT variant."
        },
    )
    output_basis_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a NumPy output basis fitted on training data. None selects the DCT variant."
        },
    )
