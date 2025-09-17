import torch
from torch import nn


class InvertedDataEmbedding(nn.Module):
    """
    Inverted Data Embedding layer for iTransformer.
    """
    def __init__(self,
                 input_len: int,
                 hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_len, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None):
        """
        Forward pass of the InvertedDataEmbedding layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features].
            inputs_timestamps (torch.Tensor): Optional tensor of timestamps of shape [batch_size, seq_len, num_timestamps]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        inputs = inputs.permute(0, 2, 1) # [batch_size, num_features, input_len]
        if inputs_timestamps is None:
            embedded = self.value_embedding(inputs)
        else:
            # the potential to take timestamps as tokens
            embedded = self.value_embedding(torch.cat([inputs, inputs_timestamps.permute(0, 2, 1)], dim=1))
        # embedded: [batch_size, num_features (+ num_timestamps), hidden_size]
        return self.dropout(embedded)

