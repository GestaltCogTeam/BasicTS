import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .utils import normalized_adj_for_gcn, normalized_adj_for_gcn_2D
import pdb

class GraphConv(nn.Module):
    """
        Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

        Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
        information from the node's neighborhood to update its own representation. This is achieved by applying a graph
        convolutional operation that combines the features of a node with the features of its neighboring nodes.

        Mathematically, the Graph Convolutional Layer can be described as follows:

            H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

        where:
            H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of 
                input features per node.
            A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
            W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
            D: The degree matrix.
    """
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()

        # Initialize the weight matrix W (in this case called `kernel`)
        # self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # nn.init.xavier_normal_(self.kernel) # Initialize the weights using Xavier initialization

        # self.kernel = nn.Parameter(torch.eye(input_dim), requires_grad=False)
        # Initialize the bias (if use_bias is True)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias) # Initialize the bias to zeros

    def forward(self, input_tensor, adj_mat):
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        B, _, _ = input_tensor.shape
        if len(adj_mat.shape) == 3:
            output = torch.bmm(adj_mat, input_tensor)

        else:
            output = torch.matmul(adj_mat, input_tensor)
        # support = torch.bmm(input_tensor, self.kernel.unsqueeze(0).expand(B, -1, -1)) # Matrix multiplication between input and weight matrix
        # output = torch.bmm(adj_mat, support) # Sparse matrix multiplication between adjacency matrix and support
        # Add the bias (if bias is not None)
        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph 
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node 
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations 
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is 
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """
    def __init__(self, hidden_dim, use_bias=False, dropout_p=0.1):
        super(GCN, self).__init__()

        # Define the Graph Convolution layers
        self.gc1 = GraphConv(hidden_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, hidden_dim, use_bias=use_bias)
        # self.gc3 = GraphConv(hidden_dim, hidden_dim, use_bias=use_bias)

        # Define the dropout layer
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """

        # num_nodes = adj_mat.shape[-1]
        # I = torch.eye(num_nodes, dtype=torch.float32, device=input_tensor.device)
        # adj_mat_backcast = normalized_adj_for_gcn(adj_mat - I)
        if len(adj_mat.shape) == 3:
            adj_mat = normalized_adj_for_gcn(adj_mat)

        else:

            adj_mat = normalized_adj_for_gcn_2D(adj_mat)

        # Perform the first graph convolutional layer
        x = self.gc1(input_tensor, adj_mat)
        # x_backcast = self.gc1(input_tensor, adj_mat_backcast)
        # x = F.relu(x) # Apply ReLU activation function
        # x = self.dropout(x) # Apply dropout regularization

        # Perform the second graph convolutional layer
        # x = self.gc2(x+input_tensor, adj_mat)
        x = self.gc2(x, adj_mat)
        # x_backcast = self.gc2(x_backcast, adj_mat_backcast)
        # x = self.gc3(x, adj_mat)

        # Apply log-softmax activation function for classification
        # return F.log_softmax(x, dim=1)
        return x #+ input_tensor


