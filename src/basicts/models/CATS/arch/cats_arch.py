import torch
from torch import Tensor, nn

from basicts.modules.norm import RevIN

from ..config.cats_config import CATSConfig
from .cats_layers import CATSDecoder


class Projection(nn.Module):
    """
    Linear projection from hidden dimension back to patch-level raw values.

    It maps each patch embedding of size `hidden_size` back to a patch of length
    `patch_len`, then flattens the patch dimension to the final temporal
    dimension.

    Constructor style:
        Projection(config)
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.linear = nn.Linear(config.hidden_size, config.patch_len)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, P, D] where
                B - batch size
                N - number of variables
                P - number of prediction patches
                D - hidden dimension

        Returns:
            [B, N, T_pred], where T_pred = P * patch_len
        """
        x = self.linear(x)  # [B, N, P, patch_len]
        x = self.flatten(x)  # [B, N, T_pred]
        return x


class DummyEmbedding(nn.Module):
    """
    Dummy patch embedding module used by CATS.

    For each variable and prediction patch index, a learnable dummy patch is
    created and encoded into the hidden space. These dummy patches are used as
    queries in the patch decoder, while the history patches act as keys/values.

    Constructor style:
        DummyEmbedding(config)
    """

    def __init__(self, config) -> None:
        super().__init__()
        # Input encoding
        self.W_P = nn.Linear(config.patch_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # Dummy Input
        self.independence = config.independence
        if self.independence:
            self.dummies = nn.Parameter(
                0.5 * torch.randn(config.pred_patch_num, config.patch_len)
            )
        else:
            self.dummies = nn.Parameter(
                0.5
                * torch.randn(
                    config.num_features, config.pred_patch_num, config.patch_len
                )
            )
        # Positional encoding
        self.PE = nn.Parameter(
            0.04 * torch.rand(config.seq_patch_num, config.hidden_size) - 0.02
        )
        # Encoder
        self.decoder = CATSDecoder(config)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x seq_patch_num x patch_len]
        bs = x.shape[0]
        n_vars = x.shape[1]
        # Input encoding
        x = self.W_P(x) + self.PE  # x: [bs x nvars x seq_patch_num x hidden_size]
        dummies = self.W_P(
            self.dummies
        )  # dummies: [bs x nvars x pred_patch_num x hidden_size]
        x = torch.reshape(
            x, (bs * n_vars, x.shape[2], x.shape[3])
        )  # x: [bs * nvars x seq_patch_num x hidden_size]

        seq_patch = self.dropout(x)  # seq_patch: [bs * nvars x patch_num x hidden_size]

        if self.independence:
            pred_patch = dummies.unsqueeze(0).repeat(bs * n_vars, 1, 1)
        else:
            pred_patch = dummies.unsqueeze(0).repeat(bs, 1, 1, 1)
            pred_patch = torch.reshape(
                pred_patch, (bs * n_vars, pred_patch.shape[2], pred_patch.shape[3])
            )  # pred_patch: [bs * nvars x pred_patch_num x hidden_size]

        # decoder
        z = self.decoder(
            seq_patch, pred_patch
        )  # z: [bs * nvars x pred_patch_num x hidden_size]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x pred_patch_num x hidden_size]
        return z


class CATS(nn.Module):
    """
    Paper: Are Self-Attentions Effective for Time Series Forecasting?
    Link: https://arxiv.org/abs/2405.16877
    Ref Official Code: https://github.com/dongbeank/CATS
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-06, "affine": False}

    def __init__(self, config: CATSConfig) -> None:
        super().__init__()

        self.config = config

        self.patch_len = config.patch_len
        self.stride = config.stride
        self.padding_patch = config.padding_patch

        self.pred_patch_num = (
            config.output_len + config.patch_len - 1
        ) // config.patch_len
        self.seq_patch_num = int(
            (config.input_len - config.patch_len) / config.stride + 1
        )

        if self.padding_patch == "end":
            self.seq_patch_num += 1
        self.config.pred_patch_num = self.pred_patch_num
        self.config.seq_patch_num = self.seq_patch_num

        # Optional padding for the temporal axis.
        if config.padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
        else:
            self.padding_patch_layer = None

        # Backbone dummy embedding + patch decoder.
        self.backbone = DummyEmbedding(self.config)
        # Projection from hidden dimension back to raw patch values.
        self.proj = Projection(self.config)

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=False)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tensor:
        """
        Forward pass of CATS.

        Args:
            inputs: [B, L_in, N, 1], input history sequence.

        Returns:
            Tensor: predictions with shape [B, L_out, N, C_out],
                where L_out = pred_len and C_out = 1 (univariate per variable).
        """

        # ---- Instance-wise normalization ----
        if self.use_revin:
            z = self.revin(inputs, "norm")
        else:
            z = inputs

        # ---- Patching along the temporal dimension ----
        if self.padding_patch_layer is not None:
            z = self.padding_patch_layer(z.transpose(1, 2))

        # Unfold into patches: [B, N, S, patch_len]
        z_patches = z.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

        # ---- Apply CATS backbone (dummy embedding + patch decoder) ----
        z_hidden = self.backbone(z_patches)  # [B, N, P, D]
        z_out = self.proj(z_hidden).transpose(1, 2)  # [B, N, T_pred >= pred_len]

        # ---- De-normalization ----
        if self.use_revin:
            prediction = self.revin(z_out, "denorm")
        else:
            prediction = z_out

        # Keep only the first `pred_len` steps.
        return prediction
