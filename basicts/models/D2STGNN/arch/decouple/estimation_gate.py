import torch
import torch.nn as nn

class EstimationGate(nn.Module):
    r"""
    The spatial gate module.
    """
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim, input_seq_len):
        super().__init__()
        self.FC1    = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.act    = nn.ReLU()
        self.FC2    = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding1, node_embedding2, T_D, D_W, X):
        B, L, N, D = T_D.shape
        spatial_gate_feat = torch.cat([T_D, D_W, node_embedding1.unsqueeze(0).unsqueeze(0).expand(B, L,  -1, -1), node_embedding2.unsqueeze(0).unsqueeze(0).expand(B, L,  -1, -1)], dim=-1)
        hidden  = self.FC1(spatial_gate_feat)
        hidden  = self.act(hidden)
        # activation
        spatial_gate = torch.sigmoid(self.FC2(hidden))[:, -X.shape[1]:, :, :]
        X   = X * spatial_gate
        return X
