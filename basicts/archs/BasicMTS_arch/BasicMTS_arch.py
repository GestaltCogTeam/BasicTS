import torch
import torch.nn as nn
from basicts.archs.BasicMTS_arch.decoder import DecoderNN

from basicts.archs.BasicMTS_arch.encoder import EncoderNN

class BasicMTS(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        print(model_args.keys())
        num_nodes  = 207
        node_dim   = 16
        temp_dim   = 16
        input_len  = 12
        input_dim  = 3
        embed_dim  = 32
        output_len = 12

        # networks
        self.encoder = EncoderNN(input_dim, embed_dim, hidden_dim=embed_dim+node_dim+temp_dim*2)
        # spatial encoding
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        # temporal encoding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, temp_dim))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, temp_dim))
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

        # regression layer
        self.decoder = DecoderNN(hidden_dim=256, out_dim=output_len)

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
        X = history_data[..., [0, 1, 2]]
        t_i_d_data   = history_data[..., 1]
        d_i_w_data   = history_data[..., 2]

        T_i_D = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]    # [B, L, N, d]
        D_i_W = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]          # [B, L, N, d]

        # NN
        H = self.encoder(X, self.node_emb, T_i_D, D_i_W)
        Y = self.decoder(H)
        return Y
