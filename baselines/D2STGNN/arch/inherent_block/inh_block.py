import math
import torch
import torch.nn as nn

from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, fk_dim=256, first=None, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        if first:
            self.pos_encoder = PositionalEncoding(
                hidden_dim, model_args['dropout'])
        else:
            self.pos_encoder = None
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args['dropout'], bias)
        # forecast
        self.forecast_block = Forecast(hidden_dim, fk_dim, **model_args)
        # backcast
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        # sub residual
        self.sub_and_norm = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, X):
        [batch_size, seq_len, num_nodes, num_feat] = X.shape
        # Temporal Model
        # RNN
        RNN_H_raw = self.rnn_layer(X)
        # Positional Encoding
        if self.pos_encoder is not None:
            RNN_H = self.pos_encoder(RNN_H_raw)
        else:
            RNN_H = RNN_H_raw
        # MultiHead Self Attention
        Z = self.transformer_layer(RNN_H, RNN_H, RNN_H)

        # forecast branch
        forecast_hidden = self.forecast_block(
            X, RNN_H_raw, Z, self.transformer_layer, self.rnn_layer, self.pos_encoder)

        # backcast branch
        Z = Z.reshape(seq_len, batch_size, num_nodes, num_feat)
        Z = Z.transpose(0, 1)
        backcast_seq = self.backcast_fc(Z)
        backcast_seq_res = self.sub_and_norm(X, backcast_seq)

        return backcast_seq_res, forecast_hidden
