import torch
from torch import nn
import pdb 
from .utils import generate_adjacent_matrix, cosine_similarity_torch, bernstein_approximation
from .gcn import GCN
import numpy as np
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=False)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=False)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.15)

        # self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = input_data + hidden #* self.resweight                           # residual
        return hidden

class EmbeddingTrainer(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, num_nodes, c_dim) -> None:
        super().__init__()

        self.node_rp_layer = nn.Linear(num_nodes, c_dim, bias=False)
        self.node_rp_layer.weight.requires_grad = False
        self.node_inv_rp_layer = nn.Linear(c_dim, num_nodes, bias=False)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.node_inv_rp_layer(self.node_rp_layer(input_data.transpose(0,1)))
        return hidden.transpose(0,1)


class NFConnection(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, n_kernel, use_bern) -> None:
        super().__init__()
      
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=n_kernel, kernel_size=(1, input_dim), bias=False)

        sequences = nn.Parameter(torch.ones(n_kernel, input_dim), requires_grad=True)
        # 多项式近似

        if use_bern:
            self.conv1.weight = nn.Parameter(torch.cat([bernstein_approximation(seq, input_dim-1).unsqueeze(0) for seq in sequences], dim=0).unsqueeze(1).unsqueeze(1))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.conv1(input_data.transpose(1,3))
        return hidden, self.conv1.weight


class SpatiaEncoder(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, hidden_dim, output_dim, num_nodes, if_hgnn, n_kernel) -> None:
        super().__init__()
    
        self.mlp = MultiLayerPerceptron(hidden_dim, hidden_dim)
        self.gcn1 = GCN(hidden_dim)
        self.projection = nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=(1, 1), bias=False)

        # self.U = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=False)
        # nn.init.orthogonal_(self.U)
        # sigma = torch.nn.Parameter(torch.ones(num_nodes), requires_grad=True)
        # self.sigma = torch.diag_embed(sigma)
        self.if_hgnn = if_hgnn
        self.n_kernel = n_kernel

        # if self.if_hgnn:
        #     self.U_h = nn.Parameter(torch.randn(n_kernel, n_kernel), requires_grad=False)
        #     nn.init.orthogonal_(self.U_h)
        #     sigma_h = torch.nn.Parameter(torch.ones(n_kernel), requires_grad=True)
        #     self.sigma_h= torch.diag_embed(sigma_h)


    def forward(self, input_data: torch.Tensor, adj: torch.Tensor, embedding: torch.Tensor, node_embedding: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """ 

        B, d, N, _ = input_data.shape

        # U = self.U.to(input_data.device)
        # # U = U / U.norm(1)
        # node_adj = torch.einsum('mn, nk -> mk', U, self.sigma.to(input_data.device))
        # node_adj = torch.einsum('mk, kn -> mn', node_adj, U.T).relu()  
        node_adj = cosine_similarity_torch(node_embedding).relu()

        if self.if_hgnn:
            # U_h = self.U_h.to(input_data.device)
            # # U = U / U.norm(1)
            # embed_adj = torch.einsum('mn, nk -> mk', U_h, self.sigma_h.to(input_data.device))
            # embed_adj = torch.einsum('mk, kn -> mn', embed_adj, U_h.T).relu()  
            embed_adj = cosine_similarity_torch(embedding).relu()
            zero_pad = torch.zeros(B, d, self.n_kernel, 1).to(input_data.device)
            input_data = torch.cat([zero_pad, input_data], dim=2)

            adj_mat = generate_adjacent_matrix(adj, embed_adj, node_adj) 

        else:

            adj_mat = node_adj#.unsqueeze(0).expand(B, -1, -1)

        hidden = self.mlp(input_data) 
        hidden_in = hidden.transpose(1,2).squeeze(-1)
        hidden_out = self.gcn1(hidden_in, adj_mat)

        if adj is not None:
            hidden_out = hidden_out[:,self.n_kernel:]

        hidden_out = self.projection(hidden_out.transpose(1,2).unsqueeze(-1))
        return hidden_out