from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STIDConfig(BasicTSModelConfig):

    """
    Config class for STID model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    input_hidden_size: int = field(default=32, metadata={"help": "Hidden size of the imput embedding."})
    intermediate_size: Optional[int] = field(default=None, metadata={"help": "Intermediate size of MLP layers. " \
                                                                  "If None, use hidden_size in STID."})
    hidden_act: str = field(default="relu", metadata={"help": "Activation function of MLP layers."})
    num_layers: int = field(default=1, metadata={"help": "Number of MLP layers."})

    if_spatial: bool = field(default=True, metadata={"help": "Whether to use spatial (feature) embedding."})
    spatial_hidden_size: int = field(default=32, metadata={"help": "Hidden size of spatial embedding."})

    if_time_in_day: bool = field(default=False, metadata={"help": "Whether to use time of day embedding."})
    if_day_in_week: bool = field(default=False, metadata={"help": "Whether to use day of week embedding."})
    num_time_in_day: int = field(default=24, metadata={"help": "Number of timestamps in a day, e.g., " \
                                                       "24 represents the 24 distinct timestamps in a day (sampled at an hourly frequency)."})
    num_day_in_week: int = field(default=7, metadata={"help": "Number of timestamps in a week, e.g., " \
                                                      "7 represents the 7 distinct timestamps in a week."})
    tid_hidden_size: int = field(default=32, metadata={"help": "Hidden size of time of day embedding."})
    diw_hidden_size: int = field(default=32, metadata={"help": "Hidden size of day of week embedding."})
