from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class HIConfig(BasicTSModelConfig):

    """
    Config class for HI model.
    """

    input_len: int = field(metadata={"help": "Input sequence length."})
    output_len: int = field(metadata={"help": "Output sequence length."})
    reverse: bool = field(default=False, metadata={"help": "If reverse the prediction of HI."})
