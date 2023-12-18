import torch
from torch import nn, optim
import torch.nn.functional as F


class embed(nn.Module):
    def __init__(self,Input_len, num_id,num_samp,IF_node):
        super(embed, self).__init__()
        self.IF_node = IF_node
        self.num_samp = num_samp
        self.embed_layer = nn.Linear(2*Input_len,Input_len)

        self.node_emb = nn.Parameter(torch.empty(num_id, Input_len))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x):

        x = x.unsqueeze(-1)
        batch_size, _, _ ,_ = x.shape
        node_emb1 = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)

        x_1 = embed.down_sampling(x, self.num_samp)
        if self.IF_node:
            x_1 = torch.cat([x_1, embed.down_sampling(node_emb1, self.num_samp)], dim=-1)

        x_2 = embed.Interval_sample(x, self.num_samp)
        if self.IF_node:
            x_2 = torch.cat([x_2, embed.Interval_sample(node_emb1, self.num_samp)], dim=-1)

        return x_1,x_2

    @staticmethod
    def down_sampling(data,n):
        result = 0.0
        for i in range(n):
            line = data[:,:,i::n,:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result

    @staticmethod
    def Interval_sample(data,n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:,:,data_len*i:data_len*(i+1),:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result