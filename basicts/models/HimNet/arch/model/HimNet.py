import torch
import torch.nn as nn
import numpy as np


class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class HimGCRU(nn.Module):
    def __init__(
        self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HimGCN(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HimGCN(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x, state, support, embeddings):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class HimEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="S",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        in_steps = x.shape[1]

        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(in_steps):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)

        # current_input: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_input, output_hidden


class HimDecoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="ST",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, xt, init_state, support, embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden


class HimNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim=3,
        output_dim=1,
        out_steps=12,
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        ycov_dim=2,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_steps = out_steps
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing

        self.encoder_s = HimEncoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            node_embedding_dim,
            meta_axis="S",
            # meta_axis=None,
        )
        self.encoder_t = HimEncoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            tod_embedding_dim + dow_embedding_dim,
            meta_axis="T",
            # meta_axis=None,
        )

        self.decoder = HimDecoder(
            num_nodes,
            output_dim + ycov_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            st_embedding_dim,
            # meta_axis=None,
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
        )
        self.st_proj = nn.Linear(self.hidden_dim, self.st_embedding_dim)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, x, y_cov, labels=None, batches_seen=None):
        tod = x[:, -1, 0, 1]
        dow = x[:, -1, 0, 2]
        tod_emb = self.tod_embedding(
            (tod * 288).long()
        )  # (batch_size, tod_embedding_dim)
        
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, dow_embedding_dim)
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)  # (B, tod+dow)

        support = torch.softmax(
            torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1
        )

        h_s, _ = self.encoder_s(x, support, self.node_embedding)  # B, T, N, hidden
        h_t, _ = self.encoder_t(x, support, time_embedding)  # B, T, N, hidden
        h_last = (h_s + h_t)[:, -1, :, :]  # B, N, hidden (last state)

        st_embedding = self.st_proj(h_last)  # B, N, st_emb_dim
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)),
            dim=-1,
        )

        ht_list = [h_last] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.out_steps):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1),
                ht_list,
                support,
                st_embedding,
            )
            go = self.out_proj(h_de)
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)

        return output


if __name__ == "__main__":
    from torchinfo import summary

    model = HimNet(num_nodes=207).cpu()
    summary(model, [[64, 12, 207, 3], [64, 12, 207, 2]], device="cpu")
