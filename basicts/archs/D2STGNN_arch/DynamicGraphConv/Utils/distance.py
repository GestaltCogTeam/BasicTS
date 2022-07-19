import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']
        self.time_slot_emb_dim  = self.hidden_dim
        self.input_seq_len      = model_args['seq_length']
        # Time Series Feature Extraction
        self.dropout    = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim= self.hidden_dim
        # Time Slot Embedding Extraction
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        # Distance Score
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim']*2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim*2)

    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)

    def forward(self, X, E_d, E_u, T_D, D_W):
        # last pooling
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()       # X->[batch_size, seq_len, num_nodes]->[batch_size, num_nodes, seq_len]
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))     # [batchsize, num_nodes, hidden_dim]
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        # distance calculation
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)                    # hidden state for calculating distance
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)                    # hidden state for calculating distance
        X  = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list
