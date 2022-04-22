import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.log import clock

class TCN(nn.Module):
    def __init__(self, input_dim, dilation_dim, skip_dim, dropout=0.3,  kernel_size=2, blocks=4, layers=2):
        super(TCN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        receptive_field = 1

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=input_dim, out_channels=dilation_dim, kernel_size=(1,kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=input_dim, out_channels=dilation_dim, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_dim, out_channels=input_dim, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_dim, out_channels=skip_dim, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(input_dim))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feedforward function of Graph WaveNet

        Args:
            history_data (torch.Tensor): shape [B, C, N, L]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        input = history_data
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        return x
