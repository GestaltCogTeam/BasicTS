from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class LightTSConfig(BasicTSModelConfig):

    """
    Config class for LightTS model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size."})
    chunk_size: int = field(default=16, metadata={"help": "Chunk (patch) size."})
