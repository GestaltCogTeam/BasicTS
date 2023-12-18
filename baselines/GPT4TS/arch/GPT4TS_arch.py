import torch
import torch.nn as nn

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT4TS(nn.Module):
    """
    Paper: One Fits All:Power General Time Series Analysis by Pretrained LM
    Link: https://arxiv.org/abs/2302.11939
    Ref Official Code: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
    """

    def __init__(self, **model_args):
        super(GPT4TS, self).__init__()
        self.is_gpt = model_args["is_gpt"]
        self.pretrain = model_args["pretrain"]
        self.seq_len = model_args["seq_len"]
        self.label_len = int(model_args["label_len"])
        self.pred_len = model_args["pred_len"]
        self.patch_size = model_args["patch_size"]
        self.stride = model_args["stride"]
        self.d_model = model_args["d_model"]
        self.d_ff = model_args["d_ff"]
        self.enc_in = model_args["enc_in"]
        self.gpt_layers = model_args["gpt_layers"]
        self.c_out = model_args["c_out"]

        self.time_of_day_size = model_args.get("time_of_day_size", None)
        self.day_of_week_size = model_args.get("day_of_week_size", None)
        self.day_of_month_size = model_args.get("day_of_month_size", None)
        self.day_of_year_size = model_args.get("day_of_year_size", None)
        self.embed = model_args["embed"]

        self.patch_num = (self.seq_len + self.pred_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if self.is_gpt:
            if self.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.

        self.enc_embedding = DataEmbedding(
                                                    model_args["dec_in"],
                                                    model_args["d_model"],
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    model_args["embed"],
                                                    model_args["num_time_features"],
                                                    model_args["dropout"])

        if self.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear = nn.Linear(self.patch_size, self.enc_in)
        self.ln = nn.LayerNorm(self.d_ff)
        self.out_layer = nn.Linear(self.d_ff, self.c_out)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
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
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = dec_out.reshape(B, -1)

        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)
