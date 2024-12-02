import torch
from torch import nn
import numpy as np 
import pdb
import time

from .mlp import MultiLayerPerceptron, SpatiaEncoder, NFConnection, EmbeddingTrainer
from .utils import link_to_onehot


class DailyCycle(torch.nn.Module):
    def __init__(self, cycle_len, num_nodes):
        super(DailyCycle, self).__init__()
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, num_nodes), requires_grad=True)

    def forward(self, index):
        return self.data[index.long()].to(index.device)

class WeeklyCycle(torch.nn.Module):
    def __init__(self, cycle_len, num_nodes):
        super(WeeklyCycle, self).__init__()
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, num_nodes), requires_grad=True)

    def forward(self, index):
        return self.data[index.long()].to(index.device)

class STHG(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.embed_sparsity = model_args["embed_sparsity"]
        self.embed_std = model_args["embed_std"]
        self.constant_c = model_args["constant"]
        self.topk = model_args["topk"]

        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.n_kernel = model_args['n_kernel']
        self.if_hgnn = model_args['if_hgnn']
        self.if_cycle = model_args['if_cycle']
        self.if_revin = model_args['if_revin']
        self.if_temporal = model_args['if_temporal']
        self.use_bern = model_args['use_bern']

        # node embeddings
        self.node_emb = self.init_emb(self.num_nodes, grad=False)
        h = self.constant_c * self.embed_sparsity * np.log(self.num_nodes / self.embed_sparsity)
        self.node_emb_trainer = EmbeddingTrainer(self.num_nodes, int(h))

        # temporal embeddings
        if self.if_temporal:
            self.time_in_day_emb = self.init_emb(self.time_of_day_size, grad=True)
            self.day_in_week_emb = self.init_emb(self.day_of_week_size, grad=True)
            self.n_feature = 3 # node tid diw # int(3 + self.topk)
        else:
            self.n_feature = 1
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=False)
        
        if self.if_hgnn:
            self.spatial_hidden_dim = self.n_feature * self.embed_dim + self.topk * self.embed_dim + self.embed_dim 
            self.temporal_hidden_dim = (self.n_feature + 1) * self.embed_dim + self.topk * self.embed_dim + self.embed_dim 
            self.feature_emb = self.init_emb(self.n_kernel)
        else:
            self.spatial_hidden_dim = self.n_feature * self.embed_dim + self.embed_dim
            self.temporal_hidden_dim = (self.n_feature + 1) * self.embed_dim + self.embed_dim
            self.feature_emb = None

        self.node_feature_connection = NFConnection(self.input_len, self.n_kernel, self.use_bern) 
        self.spatial_encoder = SpatiaEncoder(self.spatial_hidden_dim, self.embed_dim, self.num_nodes, self.if_hgnn, self.n_kernel)
        self.temporal_encoder = nn.Sequential(*[MultiLayerPerceptron(self.temporal_hidden_dim, self.temporal_hidden_dim) for _ in range(self.num_layer)])
       
        # regression
        self.temporal_regression = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.spatial_regression = nn.Conv2d(in_channels=self.temporal_hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        if self.if_cycle:
            self.dailyQueue = DailyCycle(cycle_len=self.time_of_day_size, num_nodes=self.num_nodes)
            self.weeklyQueue = WeeklyCycle(cycle_len=self.day_of_week_size, num_nodes=self.num_nodes)

        self.w_spatial = torch.nn.Parameter(torch.ones(self.num_nodes), requires_grad=True)
            
    def init_emb(self, granularity: int, embed_dim = None, grad: bool = True):
        if not embed_dim:
            emb = nn.Parameter(torch.empty(granularity, self.embed_dim), requires_grad=grad)
        else:
            emb = nn.Parameter(torch.empty(granularity, embed_dim), requires_grad=grad)
        nn.init.sparse_(emb, sparsity=self.embed_sparsity, std=self.embed_std)
        return emb

    def spatial_forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of spatial feature.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        # temporal embeddings
        tem_emb = []
        if self.if_temporal:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).long()]

            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).long()]

            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape

        time_series_emb = self.time_series_emb_layer(input_data)
        
        self.projected_node_emb = self.node_emb_trainer(self.node_emb)
        
        node_emb = []
            # expand node embeddings
        node_emb.append(self.projected_node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        if self.if_hgnn:
            connection, feature_sequence = self.node_feature_connection(input_data)
            feature_sequence = feature_sequence.squeeze()
            c_weight, c_index = torch.topk(connection, dim=1, k=self.topk) # (B, k, N, 1)
            c_weight = c_weight.squeeze(-1).transpose(1,2) # (B, N, k)
            c_index = c_index.squeeze(-1).transpose(1,2) # (B, N, k)
            corr_emb = c_weight[..., range(self.topk)].sigmoid_().unsqueeze(-1) * (self.feature_emb[c_index[..., range(self.topk)].long()])
            # corr_emb = c_weight[..., range(self.topk)].sigmoid_().unsqueeze(-1) * (feature_sequence[c_index[..., range(self.topk)].long()])

            tem_emb.append(corr_emb.reshape(batch_size, num_nodes, -1).transpose(1,2).unsqueeze(-1))
            feature_adj = link_to_onehot(c_index[..., range(self.topk)].transpose(1,2), self.n_kernel) # 输入序列的连边
        else:
            feature_adj = None
            # feature_sequence = None

        temporal_enc_in = [torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)]
        spatial_enc_in = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        spatial_enc_out = self.spatial_encoder(spatial_enc_in, feature_adj, self.feature_emb, self.projected_node_emb)
        # spatial_enc_out = self.spatial_encoder(spatial_enc_in, feature_adj, feature_sequence, self.projected_node_emb)

        temporal_enc_in.append(spatial_enc_out)

        temporal_enc_in = torch.cat(temporal_enc_in, dim=1) # (B, d, N, 1)
        temporal_enc_out = self.temporal_encoder(temporal_enc_in)

        # # regression
        forecast = self.spatial_regression(temporal_enc_out)
        return forecast.squeeze(-1)

    def temporal_forward(self, history_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of spatial feature.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        x = history_data[..., 0]

        daily_index_backcast = history_data[..., 1] * self.time_of_day_size
        daily_index_backcast = daily_index_backcast[..., 0]

        weekly_index_backcast = history_data[..., 2] * self.day_of_week_size
        weekly_index_backcast = weekly_index_backcast[..., 0]


        if self.if_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        backcast = self.dailyQueue(daily_index_backcast) + self.weeklyQueue(weekly_index_backcast)
        x = x - backcast
        forecast = self.temporal_regression(x.unsqueeze(-1)).squeeze(-1)

        daily_index_forecast = future_data[..., 1] * self.time_of_day_size
        daily_index_forecast = daily_index_forecast[..., 0]

        weekly_index_forecast = future_data[..., 2] * self.day_of_week_size
        weekly_index_forecast = weekly_index_forecast[..., 0]

        forecast = forecast + self.dailyQueue(daily_index_forecast)
        forecast = forecast + self.weeklyQueue(weekly_index_forecast)

        if self.if_revin:
            forecast = forecast * torch.sqrt(seq_var) + seq_mean
            backcast = backcast * torch.sqrt(seq_var) + seq_mean

        return backcast, forecast

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        spatial_forecast = self.spatial_forward(history_data)

        if self.if_cycle:
            temporal_backcast, temporal_forecast = self.temporal_forward(history_data, future_data)
            prediction = self.w_spatial * spatial_forecast + (1 - self.w_spatial) * temporal_forecast

        else:
            prediction = spatial_forecast
        
        return prediction.unsqueeze(-1) 


