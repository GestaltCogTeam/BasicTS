from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):

        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        A = A.to(x.device)
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        if type == 'RNN':
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'hyper':
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict([('fc1', nn.Linear((gdep + 1) * dims[0], dims[1])),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(dims[1], dims[2])),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(dims[2], dims[3]))]))

        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):

        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.beta * self.gconv(
                    h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)
        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)

        return ho
