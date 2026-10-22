from dataclasses import dataclass, field
from typing import List

from basicts.configs import BasicTSModelConfig


@dataclass
class TriformerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    input_dim: int = 1
    channels: int = 32
    patch_sizes: List[int] = field(default_factory=lambda: [7, 4, 3, 2, 2])
    memory_size: int = 5
