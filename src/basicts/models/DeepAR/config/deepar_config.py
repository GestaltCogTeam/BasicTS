from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DeepARConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    num_timestamps: int = 0
    embedding_size: int = 32
    hidden_size: int = 64
    num_layers: int = 3
    use_series_id: bool = True
    id_embedding_size: int = 32
