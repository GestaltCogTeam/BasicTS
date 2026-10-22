from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CATSConfig(BasicTSModelConfig):
    """
    Configuration class for the CATS forecasting model.
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
        default=24, metadata={"help": "Length of each extracted patch."}
    )
    stride: int = field(default=24, metadata={"help": "Stride for patch extraction."})
    padding_patch: str = field(
        default="end", metadata={"help": "Position of the time series patch."}
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
    num_layers: int = field(default=3, metadata={"help": "Number of decoder layers."})

    independence: bool = field(
        default=True,
        metadata={
            "help": "Query independence in CATS. Control whether DummyEmbedding assigns learnable parameters to each channel"
        },
    )
    qam_start: float = field(
        default=0.1,
        metadata={
            "help": "Control the start value of the mask dropout_prob for QueryAdaptiveMasking "
        },
    )
    qam_end: float = field(
        default=0.2,
        metadata={
            "help": "Control the end value of the mask dropout_prob for QueryAdaptiveMasking "
        },
    )

    # ================================================================
    # Optimization
    # ================================================================
    dropout: float = field(
        default=0.2, metadata={"help": "Dropout rate applied to multiple modules."}
    )

    # ================================================================
    # Normalization and model behaviors
    # ================================================================
    use_revin: bool = field(
        default=True, metadata={"help": "Enable RevIN normalization."}
    )
