from dataclasses import dataclass, field
from typing import Literal

from basicts.configs import BasicTSModelConfig


@dataclass
class PatchTSTConfig(BasicTSModelConfig):

    """
    Config class for PatchTST model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=8, metadata={"help": "Stride for patching."})
    padding: bool = field(default=True, metadata={"help": "Whether to pad the input sequence before patching."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    n_heads: int = field(default=1, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    attn_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for attention layers."})
    fc_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for FC layers."})
    head_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for head layers."})
    norm_type: Literal["layer_norm", "batch_norm"] = \
        field(default="layer_norm", metadata={"help": "Normalization type."})
    individual_head: bool = field(default=False, metadata={"help": "Whether to use individual head in PatchTSTHead."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    affine: bool = field(default=True, metadata={"help": "Whether to use affine transformation in RevIN."})
    subtract_last: bool = field(default=False, metadata={"help": "Whether to subtract the last element in RevIN."})
    decomp: bool = field(default=False, metadata={"help": "Whether to use decomposition."})
    moving_avg: int = field(default=25, metadata={"help": "Moving average window size for decomposition."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})
