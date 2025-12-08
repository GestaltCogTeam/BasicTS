from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class FiLMConfig(BasicTSModelConfig):

    """
    Config class for FiLM model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: list = field(default_factory = lambda: [256], metadata={"help": "Hidden size."})
    multiscale: list = field(default_factory = lambda: [1, 2, 4], metadata={"help": "Different scales for input length."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})

