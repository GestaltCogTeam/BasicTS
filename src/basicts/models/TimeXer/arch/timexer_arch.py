import torch
from torch import nn

from basicts.modules.embed import PatchEmbedding, SequenceEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import MultiHeadAttention, Seq2SeqDecoder

from ..config.timexer_config import TimeXerConfig
from .layers import FlattenHead, TimeXerEncoderLayer


class TimeXer(nn.Module):
    """
    Paper: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
    Link: https://arxiv.org/abs/2402.19072
    Official Code: https://github.com/thuml/TimeXer
    Venue:  NIPS 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, config: TimeXerConfig):
        super().__init__()
        self.num_patches = int(config.input_len // config.patch_len)
        self.num_features = config.num_features
        self.output_attentions = config.output_attentions
        self.glb_token = nn.Parameter(
            torch.randn(1, config.num_features, 1, config.hidden_size))
        # Embedding
        self.patch_embed = PatchEmbedding(
            config.hidden_size,
            config.patch_len,
            stride=config.patch_len,
            dropout=config.dropout)
        self.ex_embed = SequenceEmbedding(
            config.input_len, config.hidden_size, config.dropout)

        # TimeXer is an encoder-only model with self-attention and cross-attention layers,
        # which is actually a seq2seq decoder without causal mask.
        self.encoder = Seq2SeqDecoder(
            nn.ModuleList(
                [
                    TimeXerEncoderLayer(
                        self_attn=MultiHeadAttention(
                            config.hidden_size, config.n_heads, config.dropout),
                        cross_attn=MultiHeadAttention(
                            config.hidden_size, config.n_heads, config.dropout),
                        ffn_layer=MLPLayer(
                            config.hidden_size,
                            config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dropout=config.dropout),
                        layer_norm=(nn.LayerNorm, config.hidden_size)
                    )
                    for _ in range(config.num_layers)
                ]
            ),
            layer_norm=nn.LayerNorm(config.hidden_size)
        )

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(affine=False)

        head_nf = config.hidden_size * (self.num_patches + 1)
        self.head = FlattenHead(head_nf, config.output_len, dropout=config.dropout)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:

        """
        Forward pass of TimeXer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, input_len, num_features].
            inputs_timestamps (torch.Tensor): Input timestamps tensor of shape [batch_size, input_len, num_timestamps].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_len, num_features].
        """

        batch_size = inputs.size(0)

        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        # patching: [batch_size * num_features, num_patches, hidden_size]
        hidden_states = self.patch_embed(inputs)
        hidden_states = hidden_states.reshape(
            batch_size, self.num_features, self.num_patches, -1)
        # add global token: [batch_size * num_features, num_patches + 1, hidden_size]
        global_token_states = self.glb_token.repeat((batch_size, 1, 1, 1))
        hidden_states = torch.cat([hidden_states, global_token_states], dim=2)
        hidden_states = hidden_states.reshape(
            batch_size * self.num_features, self.num_patches + 1, -1)
        # add exogenous variables
        ex_hidden_states = self.ex_embed(inputs, inputs_timestamps)

        hidden_states, self_attn_weights, cross_attn_weights = self.encoder(
            hidden_states, ex_hidden_states, output_attentions=self.output_attentions)
        hidden_states = hidden_states.reshape(
            batch_size, self.num_features, self.num_patches + 1, -1)
        hidden_states = hidden_states.transpose(-1, -2)

        prediction = self.head(hidden_states).transpose(1, 2)

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        if self.output_attentions:
            return {"prediction": prediction,
                    "self_attn_weights": self_attn_weights,
                    "cross_attn_weights": cross_attn_weights}
        else:
            return prediction
