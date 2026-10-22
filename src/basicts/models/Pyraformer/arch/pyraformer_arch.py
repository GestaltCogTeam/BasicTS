from basicts.modules.embed.time_features import calendar_features
from ..config.pyraformer_config import PyraformerConfig
import torch
import torch.nn as nn
from basicts.modules.embed.calendar_embed import CalendarFeatureEmbedding
from .layers import (
    Bottleneck_Construct,
    Conv_Construct,
    MaxPooling_Construct,
    AvgPooling_Construct,
)
from .layers import EncoderLayer, Decoder, Predictor
from .layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(self, config: PyraformerConfig):
        super().__init__()
        self.time_of_day_size = 24
        self.day_of_week_size = 7
        self.day_of_month_size = 31
        self.day_of_year_size = 366
        self.embed = config.embed
        self.d_model = config.hidden_size
        self.window_size = config.window_size
        self.truncate = config.truncate
        if config.decoder == "attention":
            (self.mask, self.all_size) = get_mask(
                config.input_len, config.window_size, config.inner_size
            )
        else:
            (self.mask, self.all_size) = get_mask(
                config.input_len + 1, config.window_size, config.inner_size
            )
        self.decoder_type = config.decoder
        if config.decoder == "FC":
            self.indexes = refer_points(self.all_size, config.window_size)
        if config.use_tvm:
            assert len(set(self.window_size)) == 1, (
                "Only constant window size is supported."
            )
            padding = 1 if config.decoder == "FC" else 0
            q_k_mask = get_q_k(
                config.input_len + padding, config.inner_size, config.window_size[0]
            )
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        config.n_heads,
                        config.key_size,
                        config.value_size,
                        dropout=config.dropout,
                        normalize_before=False,
                        use_tvm=True,
                        q_k_mask=q_k_mask,
                        k_q_mask=k_q_mask,
                    )
                    for i in range(config.num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    EncoderLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        config.n_heads,
                        config.key_size,
                        config.value_size,
                        dropout=config.dropout,
                        normalize_before=False,
                    )
                    for i in range(config.num_layers)
                ]
            )
        # Preserve the source encoder's value/position-only embedding.
        self.enc_embedding = CalendarFeatureEmbedding(
            config.num_features,
            config.hidden_size,
            num_timestamps=config.num_timestamps,
            continuous=config.embed == "timeF",
            fixed=config.embed == "fixed",
            use_timestamps=False,
            value_bias=True,
            dropout=config.dropout,
        )
        constructors = {
            "Bottleneck_Construct": Bottleneck_Construct,
            "Conv_Construct": Conv_Construct,
            "MaxPooling_Construct": MaxPooling_Construct,
            "AvgPooling_Construct": AvgPooling_Construct,
        }
        if config.pyramid_constructor not in constructors:
            raise ValueError(
                f"Unknown pyramid_constructor: {config.pyramid_constructor}"
            )
        self.conv_layers = constructors[config.pyramid_constructor](
            config.hidden_size, config.window_size, config.bottleneck_size
        )

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)
        for i in range(len(self.layers)):
            (seq_enc, _) = self.layers[i](seq_enc, mask)
        if self.decoder_type == "FC":
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
                seq_enc.device
            )
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == "attention" and self.truncate:
            seq_enc = seq_enc[:, : self.all_size[0]]
        return seq_enc


class Pyraformer(nn.Module):
    """
    Paper: Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting
    Link: https://openreview.net/forum?id=0EXmFzUn5I
    Ref Official Code: https://github.com/ant-research/Pyraformer
    Venue: ICLR 2022
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: PyraformerConfig):
        super().__init__()
        self.time_of_day_size = 24
        self.day_of_week_size = 7
        self.day_of_month_size = 31
        self.day_of_year_size = 366
        self.embed = config.embed
        self.predict_step = config.output_len
        self.d_model = config.hidden_size
        self.input_size = config.input_len
        self.decoder_type = config.decoder
        self.channels = config.num_features
        self.encoder = Encoder(config)
        if config.decoder == "attention":
            mask = get_subsequent_mask(
                config.input_len, config.window_size, config.output_len, config.truncate
            )
            self.decoder = Decoder(config, mask)
            self.predictor = Predictor(config.hidden_size, config.num_features)
        elif config.decoder == "FC":
            self.predictor = Predictor(
                4 * config.hidden_size, config.output_len * config.num_features
            )
        self.output_len = config.output_len
        self.num_timestamps = config.num_timestamps

    def forward_xformer(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: torch.Tensor = None,
        dec_self_mask: torch.Tensor = None,
        dec_enc_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Feed forward of PyraFormer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in PyraFormer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        if self.decoder_type == "attention":
            enc_output = self.encoder(
                x_enc[:, : self.input_size], x_mark_enc[:, : self.input_size]
            )
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)
            pred = self.predictor(dec_enc)
        elif self.decoder_type == "FC":
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(
                enc_output.size(0), self.predict_step, -1
            )
        return pred

    def forward(self, inputs, inputs_timestamps=None, targets_timestamps=None):
        marks = calendar_features(
            inputs_timestamps, inputs, inputs.shape[1], self.num_timestamps
        )
        future_marks = calendar_features(
            targets_timestamps, inputs, self.output_len, self.num_timestamps
        )
        zeros = inputs.new_zeros(inputs.shape[0], self.output_len, inputs.shape[2])
        encoder = torch.cat([inputs.float(), torch.zeros_like(inputs[:, :1])], dim=1)
        encoder_marks = torch.cat([marks, future_marks[:, :1]], dim=1)
        return self.forward_xformer(encoder, encoder_marks, zeros, future_marks)
