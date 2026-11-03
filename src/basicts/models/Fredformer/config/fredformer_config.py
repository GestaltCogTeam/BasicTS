from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class FredformerConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    patch_len: int = 4
    stride: int = 4
    hidden_size: int = 24
    head_dropout: float = 0.0
    padding_patch: str = "end"
    individual: bool = False
    use_revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    cf_dim: int = 128
    cf_depth: int = 2
    cf_heads: int = 8
    cf_mlp: int = 96
    cf_head_dim: int = 32
    cf_dropout: float = 0.2
    use_nystrom: bool = True
    ablation: int = 0
    mlp_dropout: float = 0.3
    mlp_hidden_size: int = 64
    output: int = 0
