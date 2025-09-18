import torch.nn as nn

from ..decouple.residual_decomp import ResidualDecomp
from .forecast import Forecast
from .dif_model import STLocalizedConv


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, \
            use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        # sub and norm
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        # forecast
        self.forecast_branch = Forecast(
            hidden_dim, fk_dim=fk_dim, **model_args)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, X_spa, dynamic_graph, static_graph):
        Z = self.localized_st_conv(X_spa, dynamic_graph, static_graph)
        # forecast branch
        forecast_hidden = self.forecast_branch(
            X_spa, Z, self.localized_st_conv, dynamic_graph, static_graph)
        # backcast branch
        backcast_seq = self.backcast_branch(Z)
        # Residual Decomposition
        backcast_seq = backcast_seq
        X = X[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(X, backcast_seq)

        return backcast_seq_res, forecast_hidden
