from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class CycleNetConfig(BasicTSModelConfig):
    """
    Configuration class for the CycleNet forecasting model.
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
    # Model dimensions & Backbone Selection
    # ================================================================
    hidden_size: int = field(
        default=512, metadata={"help": "Hidden dimension of the transformer."}
    )
    cycle_len: int = field(
        default=24,
        metadata={
            "help": "Length of the periodic cycle (e.g., 24 for daily, 168 for weekly)."
        },
    )
    cycle_pattern: str = field(
        default="daily&weekly",
        metadata={"help": "Cycle pattern: 'daily' or 'daily&weekly'."},
    )
    model_type: str = field(
        default="mlp",
        metadata={"help": "Type of regression backbone: 'linear' or 'mlp'."},
    )

    # ================================================================
    # Optimization
    # ================================================================
    dropout: float = field(
        default=0.0, metadata={"help": "Dropout rate applied to multiple modules."}
    )
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})

    # ================================================================
    # Normalization and model behaviors
    # ================================================================
    use_revin: bool = field(
        default=True, metadata={"help": "Enable RevIN normalization."}
    )
