import numpy as np
import torch
import torch.nn as nn

from .gmsdr_cell import GMSDRCell

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.pre_k = int(model_kwargs.get('pre_k', 1))
        self.pre_v = int(model_kwargs.get('pre_v', 1))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.output_dim = int(model_kwargs.get('output_dim', 1))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.mlp = nn.Linear(self.input_dim, self.rnn_units)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.input_dim, adj_mx, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        batch = inputs.shape[0]
        x = inputs.reshape(batch, self.num_nodes, self.input_dim)
        output = self.mlp(x).view(batch, -1)
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state
        return output, torch.stack(hx_ks)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 12))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v
                       ) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hx_k: (num_layers, batch_size, pre_k, num_nodes, rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hx_ks)


class GMSDR(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.out = nn.Linear(self.rnn_units, self.decoder_model.output_dim)

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        """
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, self.num_nodes, self.rnn_units,
                           device=inputs.device)
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k)
            outputs.append(output)
        return torch.stack(outputs), hx_k

    def decoder(self, inputs, hx_k, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param inputs: (seq_len, batch_size, num_sensor * rnn_units)
        :param hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        decoder_hx_k = hx_k
        decoder_input = inputs

        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hx_k = self.decoder_model(decoder_input[t],
                                                              decoder_hx_k)
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs

    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2 = 0
        count = 0
        for key,value in base_params.items():
            if 'bias' not in key:
                loss_l2 += torch.sum(value**2)
                count += value.nelement()
        return loss_l2

    def forward(self, history_data, future_data=None, batch_seen=None, **kwargs):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        inputs = history_data.transpose(0,1).reshape(history_data.shape[1],history_data.shape[0],-1)
        encoder_outputs, hx_k = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs, hx_k, future_data, batches_seen=batch_seen)
        if batch_seen == 0:
            print(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs.transpose(0,1).reshape(history_data.shape[0],history_data.shape[1],history_data.shape[2],-1)