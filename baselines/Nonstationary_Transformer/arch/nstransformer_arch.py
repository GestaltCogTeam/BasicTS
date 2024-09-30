import torch
import torch.nn as nn
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .SelfAttention_Family import DSAttention, AttentionLayer
from .Embed import DataEmbedding
from basicts.utils import data_transformation_4_xformer
class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Nonstationary_Transformer(nn.Module):
    """
    Paper: Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
    Official Code: https://github.com/thuml/Nonstationary_Transformers
    Link: https://arxiv.org/abs/2205.14415
    """
    def __init__(self, **model_args):
        super(Nonstationary_Transformer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.label_len = int(model_args['label_len'])
        self.output_attention = model_args['output_attention']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.factor = model_args["factor"]
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.embed = model_args['embed']
        self.freq = model_args["freq"]
        self.dropout = model_args["dropout"]
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.p_hidden_dims = model_args['p_hidden_dims']
        self.p_hidden_layers = model_args['p_hidden_layers']
        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, self.factor, attention_dropout=self.dropout,
                                    output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        DSAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=self.enc_in, seq_len=self.seq_len, hidden_dims=self.p_hidden_dims,
                                     hidden_layers=self.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.enc_in, seq_len=self.seq_len,
                                       hidden_dims=self.p_hidden_dims, hidden_layers=self.p_hidden_layers,
                                       output_dim=self.seq_len)

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                        x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None, dec_self_mask: torch.Tensor = None,
                        dec_enc_mask: torch.Tensor = None) -> torch.Tensor:
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :].unsqueeze(-1), attns
        else:
            return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, D]

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=self.label_len)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction