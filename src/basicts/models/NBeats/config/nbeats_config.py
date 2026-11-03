from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class NBeatsConfig(BasicTSModelConfig):
    """
    Config class for NBEATS model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})

    model_type: str = field(
        default="generic",
        metadata={"help": "Choosing the Base Module: Generic or Decomposition."},
    )

    hidden_act: str = field(
        default="relu", metadata={"help": "Activation function of GenericBasis layers."}
    )
    num_layers: int = field(default=4, metadata={"help": "Number of NBeats layers."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    stacks: int = field(default=10, metadata={"help": "Number of NBeats layer stacks."})
    degree_of_polynomial: int = field(
        default=2, metadata={"help": "Degree of trend polynomial"}
    )
    num_of_harmonics: int = field(
        default=2, metadata={"help": "Number of harmonics for seasonality"}
    )

    # ================================================================
    # Normalization and model behaviors
    # ================================================================
    use_revin: bool = field(
        default=False, metadata={"help": "Enable RevIN normalization."}
    )
