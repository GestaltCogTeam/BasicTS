import torch
from torch import nn

from .tsformer import TSFormer
from .graphwavenet import GraphWaveNet
from .discrete_graph_learning import DiscreteGraphLearning


class STEP(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tsformer_path, short_term_len, long_term_len, tsformer_args, backend_args, dgl_args):
        super().__init__()
        
        self.short_term_len = short_term_len
        self.long_term_len = long_term_len

        self.dataset_name = dataset_name
        self.pre_trained_tsformer_path = pre_trained_tsformer_path

        # iniitalize the tsformer and backend models
        self.tsformer = TSFormer(**tsformer_args)
        self.backend = GraphWaveNet(**backend_args)

        # load pre-trained tsformer
        self.load_pre_trained_model()

        # discrete graph learning
        self.discrete_graph_learning = DiscreteGraphLearning(**dgl_args)

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tsformer_path)
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.tsformer.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        # reshape
        long_term_history = history_data     # [B, L, N, 1]
        short_term_history = history_data[:, -self.short_term_len:, :, :]

        # STEP
        batch_size, _, num_nodes, _ = short_term_history.shape

        # discrete graph learning & feed forward of TSFormer
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.discrete_graph_learning(long_term_history, self.tsformer)

        # enhancing downstream STGNNs
        hidden_states = hidden_states[:, :, -1, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj).transpose(1, 2)

        # graph structure loss coefficient
        if epoch is not None:
            gsl_coefficient = 1 / (int(epoch/6)+1)
        else:
            gsl_coefficient = 0

        prediction = y_hat.unsqueeze(-1)
        pred_adj = bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes)
        prior_adj = adj_knn
        gsl_coefficient = gsl_coefficient
        return {"prediction": prediction, "pred_adj": pred_adj, "prior_adj": prior_adj, "gsl_coefficient": gsl_coefficient}
