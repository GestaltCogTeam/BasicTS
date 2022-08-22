import torch
import torch.nn as nn
from basicts.archs.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FCLSTM(nn.Module):
    def __init__(self, input_dim, rnn_units, output_dim, horizon, num_layers, dropout=0.1):
        super(FCLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=dropout)

        #predictor
        self.end_fc = nn.Linear(self.num_layers*self.hidden_dim, self.horizon*self.output_dim)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """feedforward function of LSTM.

        Args:
            source (torch.Tensor): inputs with shape [B, L, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        B, L, N, C = history_data.shape
        # shared LSTM
        history_data = history_data.transpose(1, 2)             # [B, N, L, C]
        inputs = history_data.reshape(B*N, L, C)
        output, (h_n, c_n) = self.encoder(inputs)
        h_n = h_n.transpose(0, 1)                               # [B*N, num_layers, hidden]
        h_n = h_n.reshape(B*N, -1)                                 # [B*N, num_layers * hidden]
        h_n = h_n.view(B, N, -1)                                # [B, N, num_layers * hidden]

        # prediction
        output = self.end_fc(h_n)                               # [B, N, self.horizon*self.output_dim]
        output = output.view(B, N, self.horizon, self.output_dim)
        output = output.transpose(1, 2)                         # [B, L, N, C]

        return output
