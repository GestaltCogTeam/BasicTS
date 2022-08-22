import torch
import torch.nn as nn
from basicts.archs.AGCRN_arch.AGCRNCell import AGCRNCell
from basicts.archs.registry import ARCH_REGISTRY


"""
    Paper: Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting
    Official Code: https://github.com/LeiBAI/AGCRN
"""

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


@ARCH_REGISTRY.register()
class AGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, default_graph, embed_dim, cheb_k):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k,
                                embed_dim, num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()
    
    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        # print('*****************Model Parameter*****************')
        # only_num = False
        # if not only_num:
        #     for name, param in self.named_parameters():
        #         print(name, param.shape, param.requires_grad)
        # total_num = sum([param.nelement() for param in self.parameters()])
        # print('Total params num: {}'.format(total_num))
        # print('*****************Finish Parameter****************')

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """feedforward function of AGCRN.

        Args:
            source (torch.Tensor): inputs with shape [B, L, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        init_state = self.encoder.init_hidden(history_data.shape[0])
        output, _ = self.encoder(history_data, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output
