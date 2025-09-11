import torch
import torch.nn as nn


class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        # gated temporal conv
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        # graph conv
        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = len(self.pre_defined_graph) + \
            int(dy_graph) + int(sta_graph)
        self.num_matric = (int(use_pre) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(dy_graph) + int(sta_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim*self.num_matric, self.hidden_dim)

        # others
        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:  # staitic or predefined graph
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        # Only used in static including static hidden graph and predefined graph, but not used for dynamic graph.
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph                           # 1 order
            graph_ordered.append(k_1_order * mask)
            # e.g., order = 3, k=[2, 3]; order = 2, k=[2]
            for k in range(2, self.k_s+1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        # get st localed graph
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            # [num_nodes, kernel_size x num_nodes]
            st_local_graph.append(graph)
        # [order, num_nodes, kernel_size x num_nodes]
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        # X: [bs, seq, nodes, feat]
        # [bs, seq, num_nodes, ks, num_feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        # seq_len is changing
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        # support
        support = []
        # predefined graph
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        # dynamic graph
        if self.use_dynamic_hidden_graph:
            # k_order is caled in dynamic_graph_constructor component
            support = support + dynamic_graph
        # predefined graphs and static hidden graphs
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        # parallelize
        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        # batch_size, seq_len, num_nodes, kernel_size * hidden_dim
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        # batch_size, seq_len, kernel_size x num_nodes, hidden_dim
        X_k = out.transpose(-3, -2).reshape(batch_size,
                                            seq_len, kernel_size*num_nodes, num_feat)
        # Nx3N 3NxD -> NxD: batch_size, seq_len, num_nodes, hidden_dim
        hidden = self.gconv(support, X_k, X_0)
        return hidden
