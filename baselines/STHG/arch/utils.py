import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
import pdb
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result  # 返回函数结果
    return wrapper

# @timeit
def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

# @timeit
def link_to_onehot(mat, granularity):
    B, _, N = mat.shape # B, 1, N
    # 创建一个形状为 (64, 20, 170) 的全零张量
    one_hot_tensor = torch.zeros(B, granularity, N).to(mat.device)

    # 使用 scatter 将原始张量的值分散到 one-hot 编码张量中
    # shape:(batch, granularity+1, num node)
    return one_hot_tensor.scatter_(1, mat.type(torch.LongTensor).to(mat.device), 1)
    # return one_hot_tensor.scatter_(1, mat[:,-1,:].type(torch.LongTensor).unsqueeze(1).to(mat.device), 1)

# @timeit
def generate_adjacent_matrix(adj, adj_emb, adj_node):
    # | E*E, E*N |
    # | N*E, N*N |
    B, E, N = adj.shape
    # adj_mat = torch.zeros(B, E+N, E+N).to(adj.device)
    adj_mat_up = torch.cat([adj_emb.unsqueeze(0).expand(B, -1, -1), adj], dim=-1)
    adj_mat_bottom = torch.cat([adj.transpose(1,2), adj_node.unsqueeze(0).expand(B, -1, -1)], dim=-1)
    adj_mat = torch.cat([adj_mat_up, adj_mat_bottom], dim=1)
    return adj_mat

# @timeit # optimize by gpt
def normalized_adj_for_gcn(adj: torch.Tensor) -> torch.Tensor:
    """
    Calculate the renormalized message passing adjacency matrix for `GCN`.
    A = A + I
    return A D^{-1}

    Args:
        adj (torch.Tensor): Adjacency matrix A

    Returns:
        torch.Tensor: Renormalized message passing adjacency matrix in `GCN`.
    """
    # 如果 `adj` 是对称的，可以跳过对称化操作
    # add self loop
    device = adj.device
    num_nodes = adj.shape[1]
    
    # 使用稀疏对角矩阵来避免不必要的全尺寸矩阵创建
    # I = torch.eye(num_nodes, dtype=torch.float32, device=device)
    # adj = adj + I  # A + I

    # 计算行和的倒数 D^{-1}
    row_sum = adj.sum(dim=1)
    d_inv = torch.pow(row_sum, -1)
    
    # 避免 inf 和 nan 值
    d_inv = torch.nan_to_num(d_inv, posinf=0.0, nan=0.0)

    # 使用矩阵形式的乘法来计算 D^{-1} A
    d_mat_inv = torch.diag_embed(d_inv)
    mp_adj = torch.bmm(d_mat_inv, adj)

    return mp_adj

def normalized_adj_for_gcn_2D(adj: torch.Tensor) -> torch.Tensor:
    """
    Calculate the renormalized message passing adjacency matrix for `GCN`.
    A = A + I
    return A D^{-1}

    Args:
        adj (torch.Tensor): Adjacency matrix A

    Returns:
        torch.Tensor: Renormalized message passing adjacency matrix in `GCN`.
    """
    # 如果 `adj` 是对称的，可以跳过对称化操作
    # add self loop
    device = adj.device
    num_nodes = adj.shape[1]
    
    # 使用稀疏对角矩阵来避免不必要的全尺寸矩阵创建
    # I = torch.eye(num_nodes, dtype=torch.float32, device=device)
    # adj = adj + I  # A + I

    # 计算行和的倒数 D^{-1}
    row_sum = adj.sum(dim=0)
    d_inv = torch.pow(row_sum, -1)
    
    # 避免 inf 和 nan 值
    d_inv = torch.nan_to_num(d_inv, posinf=0.0, nan=0.0)

    # 使用矩阵形式的乘法来计算 D^{-1} A
    d_mat_inv = torch.diag_embed(d_inv)
    mp_adj = torch.mm(d_mat_inv, adj)

    return mp_adj


# 定义伯恩斯坦多项式逼近函数
def bernstein_approximation(sequence, n):
    def bernstein_basis(k, n, x):
        return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))
    
    def bernstein_poly(sequence, n, x):
        return sum(sequence[k] * bernstein_basis(k, n, x) for k in range(n + 1))
    
    return bernstein_poly(sequence, n, torch.linspace(0, 1, len(sequence)))
