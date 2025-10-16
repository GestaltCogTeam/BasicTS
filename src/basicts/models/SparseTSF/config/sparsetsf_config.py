from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SparseTSFConfig(BasicTSModelConfig):

    """
    Config class for SparseTSF model.
    """

    input_len: int = field(metadata={"help": "Input sequence length."})
    output_len: int = field(metadata={"help": "Output sequence length."})
    period_len: int = field(metadata={"help": "Period length."})
