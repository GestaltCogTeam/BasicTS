import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_conv import *
from torch.autograd import Variable
import pdb

class BigST(nn.Module):
    """
    Paper: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
    Link: https://dl.acm.org/doi/10.14778/3641204.3641217
    Official Code: https://github.com/usail-hkust/BigST?tab=readme-ov-file
    Venue: VLDB 2024
    Task: Spatial-Temporal Forecasting
    """
    def __init__(self, seq_num, in_dim, out_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                 use_residual, use_bn, use_spatial, use_long, dropout, time_of_day_size, day_of_week_size, supports=None, edge_indices=None):
        super(BigST, self).__init__()
        self.tau = tau
        self.layer_num = 3
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim
        
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.use_long = use_long
        
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.supports = supports
        
        self.time_num = time_of_day_size
        self.week_num = day_of_week_size
        
        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)
        
        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(seq_num*in_dim, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.W_1 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        for i in range(self.layer_num):
            self.linear_conv.append(linearized_conv(hid_dim*4, hid_dim*4, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim*4))
        
        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim*4*2+hid_dim+seq_num, out_dim, kernel_size=(1, 1), bias=True)
        else:
            self.regression_layer = nn.Conv2d(hid_dim*4*2, out_dim, kernel_size=(1, 1), bias=True)

    # def forward(self, x, feat=None):
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x = history_data[:, :, :, range(self.in_dim)]         # (batch_size, in_len, data_dim)
        x = x.transpose(1,2)
        # input: (B, N, T, D)
        B, N, T, D = x.size()
        
        time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]
        
        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        
        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)

        # linearized spatial convolution
        x_pool = [x] # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g) # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g) # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1) # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1) # (B, N, 1, dim)
        for i in range(self.layer_num):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)
            
            if self.use_residual:
                x = x+residual 
                
            if self.use_bn:
                x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1) # (B, dim*4, N, 1)
        
        x = self.activation(x) # (B, dim*4, N, 1)
        
        if self.use_long:
            feat = feat.permute(0, 2, 1).unsqueeze(-1) # (B, F, N, 1)
            x = torch.cat([x, feat], dim=1)
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        
        # if self.use_spatial:

        #     supports = [support.to(x.device) for support in self.supports]
        #     edge_indices = torch.nonzero(supports[0] > 0)

        #     # s_loss = spatial_loss(node_vec1_prime, node_vec2_prime, supports, edge_indices)
        #     return x.transpose(1,2).unsqueeze(-1), s_loss
        # else:
        #     return x.transpose(1,2).unsqueeze(-1), 0
        return {"prediction": x.transpose(1,2).unsqueeze(-1)
              , "node_vec1": node_vec1_prime
              , "node_vec2": node_vec2_prime
              , "supports": self.supports
              , 'use_spatial': self.use_spatial}