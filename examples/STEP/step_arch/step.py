import torch
from torch import nn

from .tsformer import TSFormer
from .graphwavenet import GraphWaveNet
from .discrete_graph_learning import DiscreteGraphLearning


class STEP(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tsformer_path, tsformer_args, backend_args, dgl_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tsformer_path = pre_trained_tsformer_path

        # tsformer and backend model args
        tsformer_args = tsformer_args
        backend_args = backend_args
        # iniitalize the tsformer and backend models
        self.tsformer = TSFormer(**tsformer_args)
        self.backend = GraphWaveNet(**backend_args)

        # load pre-trained tsformer
        self.load_pre_trained_model()

        # discrete graph learning
        self.dynamic_graph_learning = DiscreteGraphLearning(**dgl_args)

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tsformer_path)
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.tsformer.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STEP.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]
            future_data (torch.Tensor): future data
            batch_seen (int): number of batches that have been seen
            epoch (int): number of epochs

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
            torch.Tensor: the Bernoulli distribution parameters with shape [B, N, N].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
        """

        # reshape
        short_term_history = history_data     # [B, L, N, 1]
        long_term_history = long_history_data

        # STEP
        batch_size, _, num_nodes, _ = short_term_history.shape

        # discrete graph learning & feed forward of TSFormer
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dynamic_graph_learning(long_term_history, self.tsformer)

        # enhancing downstream STGNNs
        hidden_states = hidden_states[:, :, -1, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj).transpose(1, 2)

        # graph structure loss coefficient
        if epoch is not None:
            gsl_coefficient = 1 / (int(epoch/6)+1)
        else:
            gsl_coefficient = 0
        return y_hat.unsqueeze(-1), bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes), adj_knn, gsl_coefficient
