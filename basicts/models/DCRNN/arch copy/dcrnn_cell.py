import torch


class DCGRUCell(torch.nn.Module):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Codes are modified from the official repo: 
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py, 
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    Watch out the input groundtruth of decoder, which may cause bugs when you try to extend this code.
    In order to train the model on multi-GPU, we send the parameter to different gpus in the feedforward process, which might hurt the efficiency.
    """

    def __init__(self, input_dims, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        # for support in supports:
        # self._supports.append(self._build_sparse_matrix(support))
        self._supports = adj_mx
        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        input_size = (input_dims + self._num_units) * num_matrices

        # gcon params
        self.weights_output = torch.nn.Parameter(torch.empty(input_size, num_units*2))
        self.weights_update = torch.nn.Parameter(torch.empty(input_size, num_units))
        torch.nn.init.xavier_normal_(self.weights_output)
        torch.nn.init.xavier_normal_(self.weights_update)
        self.biases_output = torch.nn.Parameter(torch.empty(num_units*2))
        self.biases_update = torch.nn.Parameter(torch.empty(num_units))
        torch.nn.init.constant_(self.biases_output, 1.0)
        torch.nn.init.constant_(self.biases_update, 0.0)

    def forward(self, inputs, hx):
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._gconv(inputs, hx, output_size, bias_start=1.0, type='output'))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units, bias_start=0.0, type='update')
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs, state, output_size, bias_start, type):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(
            x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.mm(support.to(x0.device), x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.mm(support.to(x0.device), x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        # Adds for x itself.
        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        if type == 'output':
            weights = self.weights_output.to(x.device)
            biases = self.biases_output.to(x.device)
        else:
            weights = self.weights_update.to(x.device)
            biases = self.biases_update.to(x.device)
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, weights)

        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
