import torch
import torch.nn as nn
from basicts.archs.BasicMTS_arch.encoder import EncoderNN
from basicts.archs.BasicMTS_arch.decoder import DecoderNN

class BasicMTS(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        print(model_args.keys())
        self.num_nodes  = model_args['num_nodes']
        self.node_dim   = model_args['node_dim']
        self.temp_dim   = model_args['temp_dim']
        self.input_len  = model_args['input_len']
        self.input_dim  = model_args['input_dim']
        self.embed_dim  = model_args['embed_dim']
        self.output_len = model_args['output_len']

        # networks
        self.encoder = EncoderNN(self.input_dim, self.input_len, self.embed_dim, hidden_dim=self.embed_dim+self.node_dim+self.temp_dim*2)
        # spatial encoding
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        # temporal encoding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, self.temp_dim))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, self.temp_dim))
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

        # regression layer
        self.decoder = DecoderNN(hidden_dim=self.embed_dim+self.node_dim+self.temp_dim*2, out_dim=self.output_len)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction
        """
        # reshape
        history_data = history_data.transpose(1, 3)
        # normalization
        pass
        X = history_data[..., range(self.input_dim)]
        t_i_d_data   = history_data[..., 1]
        d_i_w_data   = history_data[..., 2]

        T_i_D = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]    # [B, L, N, d]
        D_i_W = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]          # [B, L, N, d]

        # NN
        H = self.encoder(X, self.node_emb, T_i_D, D_i_W)
        Y = self.decoder(H)
        return Y
