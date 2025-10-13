from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig



@dataclass
class DCRNNConfig(BasicTSModelConfig):

    """
    Config class for DCRNN model.
    """

    num_nodes: int = field(metadata={"help": "Number of nodes (variables) in multivariate time series."})
    adj_mx: list = field(metadata={"help": "Adjacency matrices."})
    seq_len: int = field(metadata={"help": "Input sequence length."})
    horizon: int = field(metadata={"help": "Prediction horizon."})
    input_dim: int = field(metadata={"help": "Input feature dimension."})

    cl_decay_steps: int = field(default=2000, metadata={"help": "CL decay steps."})
    max_diffusion_step: int = field(default=2, metadata={"help": "Max diffusion step."})
    num_rnn_layers: int = field(default=2, metadata={"help": "Number of RNN layers."})
    output_dim: int = field(default=1, metadata={"help": "Output feature dimension."})
    rnn_units: int = field(default=64, metadata={"help": "Number of RNN units."})
    use_curriculum_learning: bool = field(default=True, metadata={"help": "Whether to use curriculum learning."})
