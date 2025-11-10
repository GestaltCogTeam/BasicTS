from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class MTSMixerConfig(BasicTSModelConfig):

    """
    Config class for MTSMixer model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    temporal_hidden_size: int = field(default=512, metadata={"help": "Hidden size for temporal mixing."})
    channel_hidden_size: int = field(default=16,
                                     metadata={"help": "Hidden size for channel mixing, which is typically" \
                                               "set to be smaller than the number of features."})
    num_layers: int = field(default=2, metadata={"help": "Number of mixer layers."})
    fac_T: bool = field(default=False, metadata={"help": "If use factorized temporal mixing."})
    fac_C: bool = field(default=False, metadata={"help": "If use factorized channel mixing."})
    down_sampling: int = field(default=2, metadata={"help": "Down sampling rate."})
    use_layer_norm: bool = field(default=True, metadata={"help": "If use layer norm."})
    use_revin: bool = field(default=True, metadata={"help": "If use RevIN."})
    individual: bool = field(default=False, metadata={"help": "If use individual linear layer for each channel."})
