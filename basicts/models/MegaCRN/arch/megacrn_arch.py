import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
    
    def forward(self, x, supports):
        x_g = []    
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    """
    Paper: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting
    Link: https://aps.arxiv.org/abs/2212.05989
    Official Code: https://github.com/deepkashiwa20/MegaCRN
    Venue: AAAI 2023
    Task: Spatial-Temporal Forecasting
    Notes:
        The first version of official MegaCRN code had information leakage issues. (Refer to [1])
        In the second version of the code, the authors have modified the model structure and used a correct setting.
        However, the authors still made some mistakes in evaluating the model's prediction accuracy.
        -   First, MegaCRN calculates the metrics (masked mae, mape, rmse) in each batch and then averages them. 
            However, a correct implementation should compute metrics over the entire test dataset.
            These two implementations are very different, since the distribution of null values in each batch is very different, 
            which significantly affects the results of masked_mae, masked_mape, and masked_rmse[2]. # NOTE: Solved by this new script [5].
        -   Second, MegaCRN pads the last batch of the test dataset. 
            When we compute metrics on the test dataset, we need to remove these padded samples [3].
        In BasicTS, we avoid these mistakes based on the unified pipeline.
        The hyper-parameters of the model, the optimizer, and other tricks like gradient clip, all follow the MegaCRN's official settings. (Refer to [4])

    [1] https://github.com/deepkashiwa20/MegaCRN/issues/1
    [2] https://github.com/chaoshangcs/GTS/issues/19#issuecomment-1079932786
    [3] https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/train.py#L145
    [4] https://github.com/deepkashiwa20/MegaCRN/blob/main/model/traintest_MegaCRN.py
    [5] https://github.com/deepkashiwa20/MegaCRN/blob/main/model/traintestv1_MegaCRN.py
    """
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
    
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
    
        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
        return value, query, pos, neg
        
    def forward(self, history_data, future_data, batch_seen=None, epoch=None, **kwargs):
    # def forward(self, x, y_cov, labels=None, batches_seen=None):
        x = history_data[..., [0]]
        y_cov = future_data[..., [1]]
        labels = future_data[..., [0]]

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)    
    
        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
    
        ht_list = [h_t]*self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batch_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
    
        return {'prediction': output, 'query': query, 'pos': pos, 'neg': neg}
