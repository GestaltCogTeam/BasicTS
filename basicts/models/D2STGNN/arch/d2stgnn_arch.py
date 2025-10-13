import torch
import torch.nn as nn
import torch.nn.functional as F

from .difusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv.dy_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

from ..config.d2stgnn_config import D2STGNNConfig


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, first=False, configs: D2STGNNConfig = None):
        super().__init__()
        self.spatial_gate = EstimationGate(
            configs['node_dim'], configs['time_emb_dim'], 64, configs['seq_length'])
        self.dif_layer = DifBlock(hidden_dim, fk_dim=fk_dim, **configs)
        self.inh_layer = InhBlock(
            hidden_dim, fk_dim=fk_dim, first=first, **configs)

    def forward(self, X: torch.Tensor, dynamic_graph: torch.Tensor, static_graph, E_u, E_d, T_D, D_W):
        """decouple layer

        Args:
            X (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            E_u (torch.Parameter): node embedding E_u
            E_d (torch.Parameter): node embedding E_d
            T_D (torch.Parameter): time embedding T_D
            D_W (torch.Parameter): time embedding D_W

        Returns:
            torch.Tensor: the undecoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
        """
        X_spa = self.spatial_gate(E_u, E_d, T_D, D_W, X)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            X=X, X_spa=X_spa, dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(
            dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """
    Paper: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
    Link: https://arxiv.org/abs/2206.09112
    Official Code: https://github.com/zezhishao/D2STGNN
    Venue: VLDB 2022
    Task: Spatial-Temporal Forecasting
    """
    def __init__(self, configs: D2STGNNConfig):
        super().__init__()
        # attributes
        self._in_feat = configs['num_feat']
        self._hidden_dim = configs['num_hidden']
        self._node_dim = configs['node_dim']
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = configs['seq_length']

        self._num_nodes = configs['num_nodes']
        self._k_s = configs['k_s']
        self._k_t = configs['k_t']
        self._num_layers = configs['num_layers']
        self._time_in_day_size = configs['time_in_day_size']
        self._day_in_week_size = configs['day_in_week_size']

        configs['use_pre'] = False
        configs['dy_graph'] = True
        configs['sta_graph'] = True

        self._configs = configs

        # start embedding layer
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, configs['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, configs['time_emb_dim']))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(
            self._hidden_dim, fk_dim=self._forecast_dim, first=True, configs=configs)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim, configs=configs))

        # dynamic and static hidden graph constructor
        if configs['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **configs)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # output layer
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, configs['gap'])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['E_d']
        E_u = inputs['E_u']
        if self._configs['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._configs['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, X):
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        # [B, L, N, d]
        # In the datasets used in D2STGNN, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
        # If you use other datasets, you may need to change this line.
        T_i_D = self.T_i_D_emb[(X[:, :, :, 1] * self._time_in_day_size).type(torch.LongTensor)]
        # [B, L, N, d]
        D_i_W = self.D_i_W_emb[(X[:, :, :, 2] * self._day_in_week_size).type(torch.LongTensor)]
        # traffic signals
        X = X[:, :, :, [0]]

        return X, node_emb_u, node_emb_d, T_i_D, D_i_W

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs (torch.Tensor): inputs with shape [B, L, N].
            inputs_timestamps (torch.Tensor): timestamps with shape [B, L, D].

        Returns:
            torch.Tensor: outputs with shape [B, L, N]
        """

        inputs = inputs.unsqueeze(-1)  # B, L, N, 1
        inputs_timestamps = inputs_timestamps.unsqueeze(2).repeat(1, 1, self._num_nodes, 1)  # B, L, N, D
        X = torch.cat((inputs, inputs_timestamps), dim=-1)  # B, L, N, 1+D

        # ==================== Prepare Input Data ==================== #
        X, E_u, E_d, T_D, D_W = self._prepare_inputs(X)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(
            E_u=E_u, E_d=E_d, X=X, T_D=T_D, D_W=D_W)

        # Start embedding layer
        X = self.embedding(X)

        spa_forecast_hidden_list = []
        tem_forecast_hidden_list = []

        tem_backcast_seq_res = X
        for index, layer in enumerate(self.layers):
            tem_backcast_seq_res, spa_forecast_hidden, tem_forecast_hidden = layer(
                tem_backcast_seq_res, dynamic_graph, static_graph, E_u, E_d, T_D, D_W)
            spa_forecast_hidden_list.append(spa_forecast_hidden)
            tem_forecast_hidden_list.append(tem_forecast_hidden)

        # Output Layer
        spa_forecast_hidden = sum(spa_forecast_hidden_list)
        tem_forecast_hidden = sum(tem_forecast_hidden_list)
        forecast_hidden = spa_forecast_hidden + tem_forecast_hidden

        # regression layer
        forecast = self.out_fc_2(
            F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        # reshape
        forecast = forecast.transpose(1, 2)
        return forecast
