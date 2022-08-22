import torch
import numpy as np
import torch.nn as nn
from basicts.archs.DCRNN_arch.DCRNN_cell import DCGRUCell
from basicts.archs.registry import ARCH_REGISTRY

"""
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Ref Official Code: 
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py, 
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])
    
    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
                optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])
    
    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
                optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


@ARCH_REGISTRY.register()
class DCRNN(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 2000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, **kwargs) -> torch.Tensor:
        """feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]
            future_data (torch.Tensor, optional): future data with shape [L, B, N*C_out]
            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction wit shape [L, B, N*C_out]
        """

        encoder_hidden_state = self.encoder(history_data)
        outputs = self.decoder(encoder_hidden_state, future_data, batches_seen=batch_seen)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  #[B, L, N*C_out]
        outputs = outputs.view(B, L, self.num_nodes, self.decoder_model.output_dim)

        if not self.training:
            assert future_data == None, 'Future data should not be visible when validating/testing.'
        else:
            pass
        
        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print("Parameter Number: ".format(count_parameters(self)))
            print(count_parameters(self))
        return outputs
