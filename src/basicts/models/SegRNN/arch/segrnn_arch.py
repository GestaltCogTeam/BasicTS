from math import ceil

import torch
from torch import nn

from basicts.modules.embed import PatchEmbedding
from basicts.modules.norm import RevIN

from ..config.segrnn_config import SegRNNConfig


class SegRNN(nn.Module):
    """
    Paper: SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting
    Official Code: https://github.com/lss-1138/SegRNN
    Link: https://arxiv.org/abs/2308.11200
    Venue: arXiv
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: SegRNNConfig):
        super().__init__()

        self.output_len = config.output_len
        self.num_features = config.num_features
        self.hidden_size = config.hidden_size
        self.out_seg_num = ceil(config.output_len / config.seg_len)

        # Embedding
        pad_len = ceil(config.input_len / config.seg_len) * config.seg_len - config.input_len
        self.embedding = PatchEmbedding(
            config.hidden_size,
            config.seg_len,
            config.seg_len,
            (pad_len, 0),
            config.dropout)

        self.rnn = nn.GRU(
            self.hidden_size, self.hidden_size, num_layers=config.num_layers, batch_first=True)
        self.pos_emb = nn.Parameter(torch.randn(self.out_seg_num, self.hidden_size // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.num_features, self.hidden_size // 2))

        self.projection = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, config.seg_len)
        )
        self.revin = RevIN(affine=False, subtract_last=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SegRNN model.

        Args:
            inputs (torch.Tensor): [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: [batch_size, output_len, num_features]
        """

        batch_size = inputs.size(0)

        # norm
        inputs = self.revin(inputs, "norm")

        # embedding: [batch_size * num_features, num_patches, hidden_size]
        hidden_states = self.embedding(inputs)

        # encoding
        _, hn = self.rnn(hidden_states)

        pos_emb = torch.cat(
            [self.pos_emb.unsqueeze(0).repeat(self.num_features, 1, 1),
             self.channel_emb.unsqueeze(1).repeat(1, self.out_seg_num, 1)
             ], dim=-1).view(-1, 1, self.hidden_size).repeat(batch_size, 1, 1)

        # [1, batch_size * num_features * seg_num, hidden_size]
        _, hy = self.rnn(
            pos_emb, hn.repeat(1, 1, self.out_seg_num).view(1, -1, self.hidden_size))

        prediction = self.projection(hy)
        prediction = prediction.view(batch_size, self.num_features, -1).transpose(1, 2)

        # denorm
        prediction = self.revin(prediction, "denorm")

        return prediction[:, :self.output_len, :]
