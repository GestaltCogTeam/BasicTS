import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data
