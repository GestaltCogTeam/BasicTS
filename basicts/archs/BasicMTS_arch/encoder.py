import torch
import torch.nn as nn

from basicts.archs.BasicMTS_arch.TCN import TCN

class EncoderNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim) -> None:
        super().__init__()
        self.embed_layer = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=(1,1), bias=True)
        # TCN encoder
        self.tcn = TCN(input_dim=hidden_dim, dilation_dim=hidden_dim, skip_dim=256)

    def prepare_data(self, input_data, node_emb, t_i_d_emb, d_i_w_emb, B, L):
        # input embedding
        hidden = self.embed_layer(input_data)       # B, D, N, L
        # expand node embedding
        node_emb = node_emb.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, L)            # B, D', N, L 
        # temporal embeddings
        tmp_emb1 = t_i_d_emb.transpose(1, 3)        # B, D', N, L
        tmp_emb2 = d_i_w_emb.transpose(1, 3)        # B, D', N, L
        # concat node embedding
        hidden = torch.cat([hidden, node_emb, tmp_emb1, tmp_emb2], dim=1)   # B, D + D' + D' + D', N, 1
        return hidden

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
        input_data = input_data.transpose(1, 3)     # B, C, N, L
        
        hidden = self.prepare_data(input_data, node_emb, t_i_d_emb, d_i_w_emb, B, L)
        # TCN encoding
        hidden = self.tcn(hidden)
        return hidden
