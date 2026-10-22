from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CARDConfig(BasicTSModelConfig):
    """
    Configuration class for the CARD model.

    This config includes all parameters that are actually used inside
    the CARD architecture (CARD, CARDEncoder, CARDEncoderLayer, Attention).
    Every field corresponds directly to a model attribute to avoid runtime errors.
    """

    # ================================================================
    # Basic sequence settings
    # ================================================================
    input_len: int = field(
        default=None, metadata={"help": "Full input sequence length."}
    )
    output_len: int = field(default=None, metadata={"help": "Prediction horizon."})
    num_features: int = field(
        default=None, metadata={"help": "Number of input channels/features."}
    )

    # ================================================================
    # Patch and token structure
    # ================================================================
    patch_len: int = field(
        default=16, metadata={"help": "Length of each extracted patch."}
    )
    stride: int = field(default=8, metadata={"help": "Stride for patch extraction."})
    model_token_number: int = field(
        default=0, metadata={"help": "Number of extra learnable model tokens."}
    )

    # ================================================================
    # Transformer dimensions
    # ================================================================
    hidden_size: int = field(
        default=128, metadata={"help": "Hidden dimension of the transformer."}
    )
    intermediate_size: int = field(
        default=256, metadata={"help": "Hidden dimension of feed-forward networks."}
    )
    n_heads: int = field(default=16, metadata={"help": "Number of attention heads."})
    num_layers: int = field(default=2, metadata={"help": "Number of encoder layers."})

    # ================================================================
    # Optimization & normalization
    # ================================================================
    dropout: float = field(
        default=0.2, metadata={"help": "Dropout rate applied to multiple modules."}
    )
    momentum: float = field(
        default=0.1, metadata={"help": "Momentum for BatchNorm running statistics."}
    )
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})

    # ================================================================
    # Dynamic projection (DP) settings for channel interaction
    # ================================================================
    dp_rank: int = field(
        default=8, metadata={"help": "Low-rank projection dimension for DP."}
    )
    merge_size: int = field(default=2, metadata={"help": "Token merge size for DP."})
    alpha: float = field(
        default=0.5,
        metadata={"help": "Exponential moving average factor for DP matrix."},
    )

    # ================================================================
    # Normalization and model behaviors
    # ================================================================
    use_revin: bool = field(
        default=True, metadata={"help": "Enable RevIN normalization."}
    )
    use_statistic: bool = field(
        default=False, metadata={"help": "Enable global statistic tokens."}
    )
    use_h_loss: bool = field(
        default=True, metadata={"help": "Enable hierarchical auxiliary loss."}
    )
