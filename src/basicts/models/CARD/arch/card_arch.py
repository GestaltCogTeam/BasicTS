from typing import List, Optional

import torch
from torch import nn

from basicts.runners.callback import AddAuxiliaryLoss
from basicts.modules.norm import RevIN
from .card_loss import card_loss
from .card_layers import CARDEncoder

from ..config.card_config import CARDConfig


class CARD(nn.Module):
    """
    Paper: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting
    Link: https://arxiv.org/abs/2305.12095
    Ref Official Code: https://github.com/wxie9/CARD
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-06, "affine": False}

    _required_callbacks: List[type] = [AddAuxiliaryLoss]

    def __init__(self, config: CARDConfig):
        super().__init__()

        if (
            config.use_statistic
            and not config.use_revin
            and not config.get("_external_normalization", False)
        ):
            raise ValueError(
                "CARD use_statistic=True requires use_revin=True. "
                "Set use_statistic=False to disable RevIN."
            )
        self.config = config
        self.patch_len = config.patch_len
        self.output_len = config.output_len
        self.stride = config.stride

        patch_num = int((config.input_len - config.patch_len) / config.stride + 1)

        self.W_pos_embed = nn.Parameter(
            torch.randn(patch_num, config.hidden_size) * 1e-2
        )
        self.model_token_number = config.model_token_number

        if self.model_token_number > 0:
            self.model_token = nn.Parameter(
                torch.randn(
                    config.num_features, self.model_token_number, config.hidden_size
                )
                * 1e-2
            )

        self.total_token_number = patch_num + self.model_token_number + 1
        self.config.total_token_number = self.total_token_number

        self.W_input_projection = nn.Linear(config.patch_len, config.hidden_size)
        self.input_dropout = nn.Dropout(config.dropout)

        self.use_statistic = config.use_statistic
        self.use_h_loss = config.use_h_loss
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=False)

        self.W_statistic = nn.Linear(2, config.hidden_size)
        self.cls = nn.Parameter(torch.randn(1, config.hidden_size) * 1e-2)

        self.W_out = nn.Linear(
            (patch_num + 1 + self.model_token_number) * config.hidden_size,
            config.output_len,
        )

        self.encoder = CARDEncoder(self.config)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        normalization_stats=None,
    ) -> torch.Tensor:
        """
        Feed forward of CARD.

        Args:
            inputs: Input data with shape: [batch_size, input_len, num_features]
            targets: Future data with shape: [batch_size, output_len, num_features]

        Returns:
            Output data with shape: [batch_size, output_len, num_features]
        """

        if self.use_revin:
            z = self.revin(inputs, "norm")
            z_mean = self.revin.mean.transpose(2, 1)
            z_std = self.revin.stdev.transpose(2, 1)
        else:
            z = inputs
            if self.use_statistic:
                if normalization_stats is None:
                    raise ValueError(
                        "CARD statistic tokens require historical normalization_stats."
                    )
                z_mean, z_std = (value.transpose(2, 1) for value in normalization_stats)
        z = z.transpose(2, 1)

        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed

        if self.use_statistic:
            z_stat = torch.cat((z_mean, z_std), dim=-1)
            if z_stat.shape[-2] > 1:
                z_stat = (z_stat - torch.mean(z_stat, dim=-2, keepdims=True)) / (
                    torch.std(z_stat, dim=-2, keepdims=True) + 1e-4
                )
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2), z_embed), dim=-2)

        else:
            cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2)

        inputs = z_embed
        b, c, t, h = inputs.shape
        outputs = self.encoder(inputs)

        # de-norm
        z_out = self.W_out(outputs.reshape(b, c, -1)).transpose(2, 1)
        if self.use_revin:
            prediction = self.revin(z_out, "denorm")
        else:
            prediction = z_out

        if self.use_h_loss and targets is not None:
            aux_loss = card_loss(prediction, targets, self.output_len)

            return {"prediction": prediction, "aux_loss": aux_loss}
        return prediction
