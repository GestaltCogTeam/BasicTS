import torch
import torch.nn as nn

from basicts.archs.BasicMTS_arch.MLP import MLP_res
from basicts.archs.BasicMTS_arch.TCN import TCN

class EncoderNN(nn.Module):
    def __init__(self, input_dim, input_len, embed_dim, hidden_dim, method='MLP') -> None:
        super().__init__()
        self.method = method
        if method == 'TCN':
            # time series TCN encoder
            self.tcn = TCN(input_dim=input_dim, dilation_dim=embed_dim, skip_dim=embed_dim)
        elif method == 'MLP':
            self.embed_layer = nn.Conv2d(in_channels=input_dim*input_len, out_channels=embed_dim, kernel_size=(1,1), bias=True)
        self.mlp1 = MLP_res(hidden_dim, hidden_dim)
        self.mlp2 = MLP_res(hidden_dim, hidden_dim)

    def forward(self, input_data:torch.Tensor, node_emb:torch.Tensor, t_i_d_emb:torch.Tensor, d_i_w_emb:torch.Tensor) -> torch.Tensor:
        """feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, L, N, C]
            node_emb (torch.Tensor): node embedding with shape [N, D']
            t_i_d_emb (torch.Tensor): temporal embedding with shape [B, L, N, D]
            d_i_w_emb (torch.Tensor): temporal embedding with shape [B, L, N, D]

        Returns:
            torch.Tensor: latent representation [B, D, N, 1]
        """
        B, L, N, _ = input_data.shape
        if self.method == 'TCN':
            # TCN embedding
            input_data = input_data.transpose(1, 3).contiguous()     # B, C, N, L
            hidden = self.tcn(input_data)               # B, D, N, L
        elif self.method == 'MLP':
            # MLP embedding
            input_data = input_data.transpose(1, 2)     # B, N, L, 1
            input_data = input_data.reshape(B, N, -1).transpose(1, 2).unsqueeze(-1)
            hidden = self.embed_layer(input_data)       # B, D, N, 1
        # expand node embedding
        node_emb = node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)  # B, D', N, 1
        tem_emb  = []
        # temporal embeddings
        if t_i_d_emb is not None:
            tem_emb.append(t_i_d_emb[:, -1, :, :].transpose(1, 2).unsqueeze(-1))                     # B, D', N, 1
        if d_i_w_emb is not None:
            tem_emb.append(d_i_w_emb[:, -1, :, :].transpose(1, 2).unsqueeze(-1))                     # B, D', N, 1
        # concat node embedding
        hidden = torch.cat([hidden, node_emb] + tem_emb, dim=1)   # B, D + D' + D' + D', N, 1
        # TCN encoding
        hidden = self.mlp2(self.mlp1(hidden))
        return hidden
