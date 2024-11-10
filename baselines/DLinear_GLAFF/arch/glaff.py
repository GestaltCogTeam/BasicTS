from argparse import Namespace

import torch
from torch import nn


class Plugin(nn.Module):
    def __init__(self, **model_args):
        super(Plugin, self).__init__()
        args = Namespace(**model_args)
        self.args = args
        self.channel = args.enc_in

        self.q = args.q
        self.hist_len = args.hist_len
        self.pred_len = args.pred_len

        self.Encoder = nn.Sequential(
            nn.Linear(6, args.dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=args.dim,
                    nhead=args.head_num,
                    dim_feedforward=args.dff,
                    dropout=args.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=args.layer_num,
                norm=nn.LayerNorm(args.dim)
            ),
            nn.Linear(args.dim, self.channel)
        )

        self.MLP = nn.Sequential(
            nn.Linear(self.hist_len, args.dff),
            nn.GELU(),
            nn.Linear(args.dff, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_enc_true, x_mark_enc, x_dec_pred, x_mark_dec):
        # map
        x_enc_map = self.Encoder(x_mark_enc)
        x_dec_map = self.Encoder(x_mark_dec)

        # denormalize
        scale_true = torch.quantile(x_enc_true, self.q, 1, True) - torch.quantile(x_enc_true, 1 - self.q, 1, True)
        scale_map = torch.quantile(x_enc_map, self.q, 1, True) - torch.quantile(x_enc_map, 1 - self.q, 1, True)
        stdev = scale_true / scale_map
        means = torch.median(x_enc_true - x_enc_map * stdev, dim=1, keepdim=True)[0]
        x_enc_map = x_enc_map * stdev + means
        x_dec_map = x_dec_map * stdev + means

        # combine
        error = x_enc_true - x_enc_map
        weight = self.MLP(error.permute(0, 2, 1)).unsqueeze(1)
        x_dec = torch.stack([x_dec_map, x_dec_pred], dim=-1)
        pred = torch.sum(x_dec * weight, dim=-1)

        return pred, x_enc_map, x_dec_map
