import torch
import torch.nn as nn
from arch.Transformer_EncDec import DecoderOnly, DecoderOnlyLayer
from arch.SelfAttention_Family import FullAttention, AttentionLayer
from arch.Embed import InputEmbedding
from arch.masking import TriangularCausalMask


class Entz(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
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

        self.use_norm =model_args['use_norm']
        self.batch_size = model_args['batch_size']
        # Embedding
        self.embedding = InputEmbedding(self.d_model)

        # self.attn_pooling = AttentionPooling(self.d_model, self.n_heads)

        # Decoder-only architecture
        self.decoder = DecoderOnly(
            [
                DecoderOnlyLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.mask = TriangularCausalMask(self.batch_size, self.seq_len)

        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L1, N, 1]
        """

        B, L, N, C = history_data.shape
        history_data = history_data[:, :, :, 0]
        
        # [B*N, L]
        x = history_data.permute(0, 2, 1).reshape(-1, L)
        
        # [B*N, L, d]
        x = self.embedding(x.unsqueeze(-1))

        # [B*N, L, d]
        x = self.decoder(x)
        x = self.projector(x)
        x = x.reshape(-1, )

        return prediction.unsqueeze(-1)
    
    def generate(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
