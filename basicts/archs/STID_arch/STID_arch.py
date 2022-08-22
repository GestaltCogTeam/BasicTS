import torch
import torch.nn as nn
from basicts.archs.STID_arch.MLP import MLP_res
from basicts.archs.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class STID(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes  = model_args['num_nodes']
        self.node_dim   = model_args['node_dim']
        self.input_len  = model_args['input_len']
        self.input_dim  = model_args['input_dim']
        self.embed_dim  = model_args['embed_dim']
        self.output_len = model_args['output_len']
        self.num_layer  = model_args['num_layer']
        self.temp_dim_tid   = model_args['temp_dim_tid']
        self.temp_dim_diw   = model_args['temp_dim_diw']

        self.if_T_i_D = model_args['if_T_i_D']
        self.if_D_i_W = model_args['if_D_i_W']
        self.if_node  = model_args['if_node']

        # spatial embeddings
        if self.if_node:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_T_i_D:
            self.T_i_D_emb  = nn.Parameter(torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.T_i_D_emb)
        if self.if_D_i_W:
            self.D_i_W_emb  = nn.Parameter(torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.D_i_W_emb)

        # embedding layer 
        self.time_series_emb_layer = nn.Conv2d(in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim*int(self.if_node)+self.temp_dim_tid*int(self.if_D_i_W) + self.temp_dim_diw*int(self.if_T_i_D)
        self.encoder = nn.Sequential(*[MLP_res(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1,1), bias=True)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        X = history_data[..., range(self.input_dim)]
        t_i_d_data   = history_data[..., 1]
        d_i_w_data   = history_data[..., 2]

        if self.if_T_i_D:
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]    # [B, N, D]
        else:
            T_i_D_emb = None
        if self.if_D_i_W:
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]          # [B, N, D]
        else:
            D_i_W_emb = None

        # time series embedding
        B, L, N, _ = X.shape
        X = X.transpose(1, 2).contiguous()                      # B, N, L, 1
        X = X.view(B, N, -1).transpose(1, 2).unsqueeze(-1)      # B, D, N, 1
        time_series_emb = self.time_series_emb_layer(X)         # B, D, N, 1

        node_emb = []
        if self.if_node:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # B, D, N, 1
        # temporal embeddings
        tem_emb  = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction
