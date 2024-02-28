import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, end_dim, num_layer, dropout, horizon):
        """Init LSTM.

        Args:
            input_dim (int): number of input features.
            embed_dim (int): dimension of the input embedding layer (a linear layer).
            hidden_dim (int): hidden size in LSTM.
            end_dim (int): hidden dimension of the output linear layer.
            num_layer (int): number of layers in LSTM.
            dropout (float): dropout rate.
            horizon (int): number of time steps to be predicted.
        """
        super(LSTM, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=input_dim, 
                                    out_channels=embed_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hidden_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of LSTM.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        x = history_data.transpose(1, 3)
        b, c, n, l = x.shape

        x = x.transpose(1,2).reshape(b*n, c, 1, l)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, l, 1).transpose(1, 2)
        return x
