from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class WaveNetConfig(BasicTSModelConfig):
    """Config for channel-independent dilated WaveNet forecasting."""

    input_len: int = field(metadata={"help": "Input sequence length."})
    output_len: int = field(metadata={"help": "Output sequence length."})
    num_features: int = field(default=1, metadata={"help": "Number of variates."})
    in_dim: int = field(default=1, metadata={"help": "Input channels per variate."})
    residual_channels: int = field(
        default=16, metadata={"help": "Residual block width."}
    )
    dilation_channels: int = field(default=16, metadata={"help": "Dilated conv width."})
    skip_channels: int = field(default=64, metadata={"help": "Skip-connection width."})
    end_channels: int = field(default=128, metadata={"help": "Decoder head width."})
    kernel_size: int = field(
        default=12, metadata={"help": "Temporal conv kernel size."}
    )
    blocks: int = field(default=6, metadata={"help": "Number of WaveNet blocks."})
    layers: int = field(default=3, metadata={"help": "Layers per block."})
