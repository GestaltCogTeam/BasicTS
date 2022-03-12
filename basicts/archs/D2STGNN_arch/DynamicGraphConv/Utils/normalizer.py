import torch
import torch.nn as nn
from basicts.utils.misc import remove_nan_inf

class Normalizer(nn.Module):
    def __init__(self, method, **model_args):
        super().__init__()
        self.method = method
        assert self.method == 'transition' or self.method == None
    
    def _norm(self, graph):
        if self.method == 'transition':
            degree  = torch.sum(graph, dim=2)
            degree  = remove_nan_inf(1 / degree)
            degree  = torch.diag_embed(degree)
            P       = torch.bmm(degree, graph)
            return P
        elif self.method == None:
            return graph
        else:
            raise Exception("Error")

    def forward(self, adj):
        return [self._norm(_) for _ in adj]

class MultiOrder(nn.Module):
    def __init__(self, method, order=2, **model_args):
        super().__init__()
        self.method = method
        self.order  = order

    def _multi_order(self, graph):
        if self.method == 'multiplication':
            graph_ordered = []
            k_1_order = graph               # 1 order
            mask = torch.eye(graph.shape[1]).to(graph.device)
            mask = 1 - mask
            graph_ordered.append(k_1_order * mask)
            for k in range(2, self.order+1):     # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
                k_1_order = torch.matmul(k_1_order, graph)
                graph_ordered.append(k_1_order * mask)
            return graph_ordered
        elif self.method == 'cheb':
            first_laplacian = torch.zeros_like(graph)
            second_laplacian = graph
            third_laplacian = 2 * torch.matmul(graph, second_laplacian) - first_laplacian
            forth_laplacian = 2 * torch.matmul(graph, third_laplacian ) - second_laplacian
            graph_ordered  = [second_laplacian, third_laplacian, forth_laplacian]
            return graph_ordered[:self.order]
        else:
            raise Exception("Error")

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]