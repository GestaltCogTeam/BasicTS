import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .dgcrn_layer import *


class DGCRN(nn.Module):
    """
    Paper: Dynamic Graph Convolutional Recurrent Network for Trafﬁc Prediction: Benchmark and Solution
    Official Code: https://github.com/tsinghua-fib-lab/Traffic-Benchmark
    Link: https://arxiv.org/abs/2104.14917
    Venue: ACM TKDD 2023
    Task: Spatial-Temporal Forecasting
    """
    def __init__(self, gcn_depth, num_nodes, predefined_A=None, dropout=0.3, subgraph_size=20, node_dim=40, middle_dim=2, seq_length=12, in_dim=2, list_weight=[0.05, 0.95, 0.95], tanhalpha=3, cl_decay_steps=4000, rnn_size=64, hyperGNN_dim=16):
        super(DGCRN, self).__init__()
        self.output_dim = 1

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A

        self.seq_length = seq_length

        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(self.num_nodes)

        self.rnn_size = rnn_size
        self.in_dim = in_dim

        self.hidden_size = self.rnn_size

        dims_hyper = [self.hidden_size + in_dim,
                      hyperGNN_dim, middle_dim, node_dim]

        self.GCN1_tg = gcn(dims_hyper, gcn_depth,
                           dropout, *list_weight, 'hyper')

        self.GCN2_tg = gcn(dims_hyper, gcn_depth,
                           dropout, *list_weight, 'hyper')

        self.GCN1_tg_de = gcn(dims_hyper, gcn_depth,
                              dropout, *list_weight, 'hyper')

        self.GCN2_tg_de = gcn(dims_hyper, gcn_depth,
                              dropout, *list_weight, 'hyper')

        self.GCN1_tg_1 = gcn(dims_hyper, gcn_depth,
                             dropout, *list_weight, 'hyper')

        self.GCN2_tg_1 = gcn(dims_hyper, gcn_depth,
                             dropout, *list_weight, 'hyper')

        self.GCN1_tg_de_1 = gcn(dims_hyper, gcn_depth,
                                dropout, *list_weight, 'hyper')

        self.GCN2_tg_de_1 = gcn(dims_hyper, gcn_depth,
                                dropout, *list_weight, 'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        self.alpha = tanhalpha
        self.k = subgraph_size
        dims = [in_dim + self.hidden_size, self.hidden_size]

        self.gz1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2 = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

        self.gz1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = gcn(dims, gcn_depth, dropout, *list_weight, 'RNN')

        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    def preprocessing(self, adj, predefined_A):
        adj = adj + torch.eye(self.num_nodes).to(adj.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def step(self, input, Hidden_State, Cell_State, predefined_A, type='encoder', i=None):
        x = input
        x = x.transpose(1, 2).contiguous()

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.num_nodes, self.hidden_size)), 2)

        if type == 'encoder':
            filter1 = self.GCN1_tg(
                hyper_input, predefined_A[0]) + self.GCN1_tg_1(hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg(
                hyper_input, predefined_A[0]) + self.GCN2_tg_1(hyper_input, predefined_A[1])

        if type == 'decoder':
            filter1 = self.GCN1_tg_de(
                hyper_input, predefined_A[0]) + self.GCN1_tg_de_1(hyper_input, predefined_A[1])
            filter2 = self.GCN2_tg_de(
                hyper_input, predefined_A[0]) + self.GCN2_tg_de_1(hyper_input, predefined_A[1])

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - \
            torch.matmul(nodevec2, nodevec1.transpose(2, 1))

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_A[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_A[1])

        Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size)

        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = torch.sigmoid(self.gz1(combined, adp) +
                              self.gz2(combined, adpT))
            r = torch.sigmoid(self.gr1(combined, adp) +
                              self.gr2(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        elif type == 'decoder':
            z = torch.sigmoid(self.gz1_de(combined, adp) +
                              self.gz2_de(combined, adpT))
            r = torch.sigmoid(self.gr1_de(combined, adp) +
                              self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1_de(
                temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + \
            torch.mul(1 - z, Cell_State)

        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(-1, self.hidden_size)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,**kwargs) -> torch.Tensor:
        """Feedforward function of DGCRN.

        Args:
            history_data (torch.Tensor): historical data with shape [B, L, N, C].
            future_data (torch.Tensor, optional): ground truth. Defaults to None.
            batch_seen (int, optional): batch num. Defaults to None.
            task_level (int, optional): curriculum learning level. Defaults to 12.

        Returns:
            torch.Tensor: prediction with shape [B, L, N, 1]
        """
        task_level = kwargs["task_level"]
        input = history_data.transpose(1, 3)
        ycl = future_data.transpose(1, 3)

        self.idx = self.idx.to(input.device)

        predefined_A = self.predefined_A
        x = input

        batch_size = x.size(0)
        Hidden_State, Cell_State = self.initHidden(
            batch_size * self.num_nodes, self.hidden_size)
        Hidden_State = Hidden_State.to(input.device)
        Cell_State = Cell_State.to(input.device)

        outputs = None
        for i in range(self.seq_length):
            Hidden_State, Cell_State = self.step(
                x[..., i].squeeze(-1), Hidden_State, Cell_State, predefined_A, 'encoder', i)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        go_symbol = torch.zeros(
            (batch_size, self.output_dim, self.num_nodes)).to(input.device)
        timeofday = ycl[:, [1], :, :]

        decoder_input = go_symbol

        outputs_final = []

        for i in range(task_level):
            try:
                decoder_input = torch.cat(
                    [decoder_input, timeofday[..., i]], dim=1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)
            Hidden_State, Cell_State = self.step(
                decoder_input, Hidden_State, Cell_State, predefined_A, 'decoder', None)

            decoder_output = self.fc_final(Hidden_State)

            decoder_input = decoder_output.view(
                batch_size, self.num_nodes, self.output_dim).transpose(1, 2)
            outputs_final.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batch_seen):
                    decoder_input = ycl[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(
            batch_size, self.num_nodes, task_level, self.output_dim).transpose(1, 2)

        ramdom_predict = torch.zeros(batch_size, self.seq_length - task_level,
                                     self.num_nodes, self.output_dim).to(outputs_final.device)
        outputs_final = torch.cat([outputs_final, ramdom_predict], dim=1)

        return outputs_final

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size))

            nn.init.orthogonal_(Hidden_State)
            nn.init.orthogonal_(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
