import torch
import torch.nn as nn

class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        # self.dropouts = nn.ModuleList()
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else: x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x
