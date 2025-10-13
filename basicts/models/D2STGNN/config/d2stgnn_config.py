from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig



@dataclass
class D2STGNNConfig(BasicTSModelConfig):

    """
    Config class for D2STGNN model.
    """

    num_nodes: int = field(metadata={"help": "Number of nodes (variables) in multivariate time series."})
    adjs: list = field(default_factory=list, metadata={"help": "Adjacency matrices."})
    num_hidden: int = field(default=32, metadata={"help": "Number of hidden units."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    seq_length: int = field(default=12, metadata={"help": "Input sequence length."})
    k_t: int = field(default=3, metadata={"help": "Temporal kernel size."})
    k_s: int = field(default=2, metadata={"help": "Spatial kernel size."})
    gap: int = field(default=3, metadata={"help": "Gap size."})
    num_feat: int = field(default=1, metadata={"help": "Number of input features."})
    num_layers: int = field(default=5, metadata={"help": "Number of layers."})
    node_dim: int = field(default=10, metadata={"help": "Node embedding dimension."})
    time_emb_dim: int = field(default=10, metadata={"help": "Time embedding dimension."})
    time_in_day_size: int = field(default=288, metadata={"help": "Number of time slots in a day."})
    day_in_week_size: int = field(default=7, metadata={"help": "Number of days in a week."})
