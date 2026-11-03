from dataclasses import dataclass, field
from typing import List

from basicts.configs import BasicTSModelConfig


@dataclass
class ModernTCNConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    stem_ratio: int = 6
    downsample_ratio: int = 2
    ffn_ratio: int = 8
    num_blocks: List[int] = field(default_factory=lambda: [3])
    large_kernel_sizes: List[int] = field(default_factory=lambda: [51])
    small_kernel_sizes: List[int] = field(default_factory=lambda: [5])
    dims: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    depthwise_dims: List[int] = field(default_factory=lambda: [256] * 4)
    small_kernel_merged: bool = False
    dropout: float = 0.8
    head_dropout: float = 0.2
    use_multi_scale: bool = False
    use_revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    patch_size: int = 8
    patch_stride: int = 4
    decomposition: bool = True
    moving_avg: int = 25
    individual: bool = False
