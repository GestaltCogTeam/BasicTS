from dataclasses import dataclass, field
from typing import List

from basicts.configs import BasicTSModelConfig


@dataclass
class NHiTSConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    output_size: int = 1
    n_blocks: List[int] = field(default_factory=lambda: [1, 1, 1])
    n_layers: List[int] = field(default_factory=lambda: [2, 2, 2])
    hidden_sizes: List[List[int]] = field(default_factory=lambda: [[512, 512]] * 3)
    pooling_sizes: List[int] = field(default_factory=lambda: [8, 8, 8])
    downsample_frequencies: List[int] = field(default_factory=lambda: [24, 12, 1])
    pooling_mode: str = "max"
    interpolation_mode: str = "linear"
    dropout: float = 0.0
    hidden_act: str = "ReLU"
    initialization: str = "lecun_normal"
    batch_normalization: bool = False
    shared_weights: bool = False
    naive_level: bool = True
