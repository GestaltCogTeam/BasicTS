#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg

def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:
    """Calculate yymmetric normalized laplacian. 
    Assuming unnormalized laplacian matrix is `L = D - A`,
    then symmetric normalized laplacian matrix is: 
    `L^{Sym} =  D^-1/2 L D^-1/2 =  D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2`
    For node `i` and `j` where `i!=j`, L^{sym}_{ij} <=0.

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Symmetric normalized laplacian L^{Sym}
    """
    adj                                 = sp.coo_matrix(adj)
    D                                   = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()    # diagonals of D^{-1/2}
    D_inv_sqrt[np.isinf(D_inv_sqrt)]    = 0.
    matrix_D_inv_sqrt                   = sp.diags(D_inv_sqrt)   # D^{-1/2}
    symmetric_normalized_laplacian      = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo() 
    return symmetric_normalized_laplacian

def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:
    """Re-scaled the eigenvalue to [-1, 1] by scaled the normalized laplacian matrix for chebyshev pol.
    According to `2017 ICLR GCN`, the lambda max is set to 2, and the graph is set to undirected.
    Note that rescale the laplacian matrix is equal to rescale the eigenvalue matrix.
    `L_{scaled} = (2 / lambda_max * L) - I`

    Args:
        adj (np.ndarray): Adjacent matrix A
        lambda_max (int, optional): Defaults to 2.
        undirected (bool, optional): Defaults to True.

    Returns:
        np.matrix: The rescaled laplacian matrix.
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L       = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L       = sp.csr_matrix(L)
    M, _    = L.shape
    I       = sp.identity(M, format='csr', dtype=L.dtype)
    L_res   = (2 / lambda_max * L) - I
    return L_res

def symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """ Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """

    # add self loop
    adj         = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj         = sp.coo_matrix(adj)
    rowsum      = np.array(adj.sum(1))
    d_inv_sqrt  = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt  = sp.diags(d_inv_sqrt)
    mp_adj          = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32)
    return mp_adj

def transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Transition matrix P
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P
