from __future__ import division
import torch
import torch.nn as nn


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = x.transpose(1,3)
        x = torch.einsum('ncwl,ncvw->ncvl',(x,A))
        x=x.transpose(1,3)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class dynamicGCN(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha,K,num_nodes,dimension):
        super(dynamicGCN, self).__init__()
        self.nconv = nconv()
        self.mlp = linear( 2*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.K =K
        self.conv_adj_current = nn.Conv2d(1,1,(1,num_nodes))
        self.conv_adj_basis = nn.Conv2d(1, 1, (1, num_nodes))
        self.Linear_current = nn.Linear(num_nodes,dimension)
        self.Linear_basis = nn.Linear(num_nodes,dimension)


    def forward(self,x,U,V,weight,sigma):
        sigma_diag = torch.stack([torch.diag(var) for var in sigma], dim=0)
        inverse_coordinate = torch.einsum('kn,BTnd->BTkd', V.transpose(1, 0), x.transpose(3,1))
        multiplication_matrix = torch.einsum('Mck,BTkd->BTMcd', sigma_diag, inverse_coordinate)
        dynamic_multiplication = torch.einsum('BTM,BTMcd->BTcd', weight, multiplication_matrix)
        origin_coordinate = torch.einsum('nc,BTcd->BTnd', U, dynamic_multiplication).transpose(3,1)
        out = [x]
        h = self.alpha * x + (1-self.alpha)* origin_coordinate
        out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


