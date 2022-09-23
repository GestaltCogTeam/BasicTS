import math
import torch
from torch import nn
import torch.nn.functional as F

def batch_cosine_similarity(x, y):
    # 计算分母
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    # 计算分子
    l2_z = torch.matmul(x, y.transpose(1, 2))
    # cos similarity affinity matrix 
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj

def batch_dot_similarity(x, y):
    _1 = y[:, [-1], :]
    _2 = x
    # TODO add QKV parameter here
    QKT = torch.bmm(x, y.transpose(-1, -2)) / math.sqrt(x.shape[2])
    W = torch.softmax(QKT, dim=-1)
    return W

def batch_bi_linear(x, y, FC):
    W = torch.matmul(x, FC).bmm(y.transpose(-1, -2))
    return W

def batch_perceptron(x, y, FC1, FC2, attn):
    B, L, D = x.shape
    x = torch.matmul(x, FC1)            # B', L, D
    y = torch.matmul(y, FC2)
    W = torch.tanh(x + y).matmul(attn)          # B, L, L, 1 
    W = W.transpose(-1, -2)
    return W

class Similarity(nn.Module):
    def __init__(self, mode='', **model_args):
        super().__init__()
        self.mode = mode
        assert mode in ['cosine', 'dot', 'bi-linear', 'perceptron']
        if mode == 'bi-linear':
            self.FC = nn.Parameter(torch.zeros(model_args['out_channel'], model_args['out_channel']))
            nn.init.kaiming_uniform_(self.FC, a=math.sqrt(5))       # TODO other
        elif mode == 'perceptron':
            self.FC1 = nn.Parameter(torch.zeros(model_args['out_channel'], model_args['out_channel']))
            self.FC2 = nn.Parameter(torch.zeros(model_args['out_channel'], model_args['out_channel']))
            self.att = nn.Parameter(torch.zeros(model_args['out_channel'], 1))
            nn.init.kaiming_uniform_(self.FC1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.FC2, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.att, a=math.sqrt(5))

    def forward(self, query, keys):
        """
        Calculate similarity.
        TODO there are some waste of calculation.

        Args:
            keys: B, N, L/P, D
            query: B, N, 1, D

        return:
            sim: B, N, L/P
        """
        B, N, L_P, D = keys.shape
        query = query.view(B*N, 1, D)
        keys = keys.view(B*N, L_P, D)     # B', L, D
        if self.mode == 'cosine':
            sim = batch_cosine_similarity(query, keys)
        elif self.mode == 'dot':
            sim = batch_dot_similarity(query, keys)
        elif self.mode == 'bi-linear':
            sim = batch_bi_linear(query, keys, self.FC)
        elif self.mode == 'perceptron':
            sim = batch_perceptron(query, keys, self.FC1, self.FC2, self.att)
        else:
            raise Exception("Error")
        sim = sim.view(B, N, 1, L_P)
        sim = sim.transpose(-1, -2)
        return sim

if __name__ == "__main__":
    model_args = {}
    model_args['out_channel'] = 36
    mode = 'dot'
    keys = torch.randn(64, 207, 48, 36)
    # query = torch.randn(1, 1, 1, 36)
    query = keys[:, :, [-1], :]
    Sim = Similarity(mode, **model_args)
    sim = Sim(query, keys)
    print(sim)