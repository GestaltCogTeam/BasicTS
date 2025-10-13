from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class AGCRNConfig(BasicTSModelConfig):

    """
    Config class for STID model.
    """

    num_nodes: int = field(metadata={"help": "Number of nodes (variables) in multivariate time series."})
    input_dim: int = field(metadata={"help": "Input feature dimension."})
    output_lens: int = field(metadata={"help": "Forecasting horizon (output sequence length)."})
    rnn_units: int = field(default=64, metadata={"help": "Hidden size of the GRU layer."})
    output_dim: int = field(default=1, metadata={"help": "Output feature dimension."})
    num_layers: int = field(default=2, metadata={"help": "Number of AGCRN layers."})
    embed_dim: int = field(default=10, metadata={"help": "Dimension of node embeddings."})
    cheb_k: int = field(default=2, metadata={"help": "Chebyshev filter size."})
