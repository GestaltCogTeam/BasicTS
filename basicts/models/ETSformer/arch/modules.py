import torch.nn as nn
import torch.nn.functional as F


class ETSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                              kernel_size=3, padding=2, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x,):

        x = self.conv(x.permute(0,2,1))[..., :-2]

        return self.dropout(x.transpose(1,2))


class Feedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)
