
import torch
from torch import nn
from .embed_block import embed
from .TVA_block import TVA_block_att
from .decoder_block import TVADE_block
from .revin import RevIN

class DSFormer(nn.Module):
    def __init__(self, Input_len, out_len, num_id, num_layer, dropout, muti_head, num_samp, IF_node):
        """
        Input_len: History length
        out_len：future length
        num_id：number of variables
        num_layer：number of layer. 1 or 2
        muti_head：number of muti_head attention. 1 to 4
        dropout：dropout. 0.15 to 0.3
        num_samp：muti_head subsequence. 2 or 3
        IF_node:Whether to use node embedding. True or False
        """
        super(DSFormer, self).__init__()

        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp

        ### embed and encoder
        self.RevIN = RevIN(num_id)
        self.embed_layer = embed(Input_len,num_id,num_samp,IF_node)
        self.encoder = TVA_block_att(self.inputlen,num_id,num_layer,dropout, muti_head,num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])

        ### decorder
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(in_channels = self.inputlen, out_channels=out_len, kernel_size=1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # Input [B,H,N,1]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length
        x = history_data[:, :, :, 0]
        ### embed
        x = self.RevIN(x,'norm').transpose(-2,-1)
        x_1, x_2 = self.embed_layer(x)

        ### encoder
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        ### decorder
        x = self.decoder(x)
        x = self.output(x.transpose(-2,-1))
        x = self.RevIN(x, 'denorm')
        x = x.unsqueeze(-1)
        return x
