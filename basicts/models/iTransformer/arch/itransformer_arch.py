import torch
from torch import nn

from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import (MultiHeadSelfAttention,
                                         TransformerBlock)

from ..config.itransformer_config import iTransformerConfig
from .embed import InvertedDataEmbedding


class iTransformer(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting, Time Series Classification
    """
    def __init__(self, config: iTransformerConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.output_attention = config.output_attention
        self.num_features = config.num_features
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.intermediate_size = config.intermediate_size
        self.dropout = config.dropout
        self.activation = config.activation
        self.num_layers = config.num_layers
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(self.num_features, affine=False)

        # Embedding
        self.enc_embedding = InvertedDataEmbedding(self.input_len, self.hidden_size, self.dropout)

        # Encoder-only architecture
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    MultiHeadSelfAttention(self.hidden_size, self.n_heads, self.dropout),
                    MLPLayer(
                        self.hidden_size,
                        self.intermediate_size,
                        hidden_act=self.activation,
                        dropout=self.dropout),
                    layer_norm=(nn.LayerNorm, {"normalized_shape": self.hidden_size}),
                    norm_position="post"
                ) for _ in range(self.num_layers)
            ]
        )
        self.post_norm = nn.LayerNorm(self.hidden_size)
        self.projector = nn.Linear(self.hidden_size, self.output_len)

        # elif self.task_name == 'classification':
        #     self.num_classes = model_args['num_classes']
        #     self.act = F.gelu
        #     self.dropout = nn.Dropout(self.dropout)
        #     self.projector = nn.Linear(self.d_model * self.enc_in, self.num_classes)
        # else:
        #     raise ValueError(f"Task name {self.task_name} is not supported.")

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        hidden_states = self.enc_embedding(inputs, inputs_timestamps)

        attn_weights = []
        for layer in self.encoder:
            hidden_states, attns, _, _ = layer(hidden_states)
            if self.output_attention:
                attn_weights.append(attns)

        hidden_states = self.post_norm(hidden_states)
        prediction = self.projector(hidden_states).transpose(1, 2)
        prediction = prediction[..., :self.num_features]

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        if self.output_attention:
            return {"prediction": prediction, "attn_weights": attn_weights}
        else:
            return prediction
