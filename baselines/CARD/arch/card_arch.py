import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from basicts.utils import data_transformation_4_xformer
from .attention import Attenion, Transpose

from argparse import Namespace

import pdb 

class CARD(nn.Module):
    """
    Paper: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting
    Link: https://arxiv.org/abs/2305.12095
    Official Code: https://github.com/thuml/TimeXer
    Venue:  ICLR 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **config):
        super().__init__()
        config = Namespace(**config)
        self.patch_len  = config.patch_len
        self.pred_len = config.pred_len
        self.stride = config.stride
        self.d_model = config.d_model
        patch_num = int((config.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num,config.d_model)*1e-2)
        self.model_token_number = config.model_token_number
        
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(config.enc_in,self.model_token_number,config.d_model)*1e-2)
        
        self.total_token_number = (self.patch_num  + self.model_token_number + 1)
        config.total_token_number = self.total_token_number
        
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)  
        self.input_dropout  = nn.Dropout(config.dropout) 
        
        self.use_statistic = config.use_statistic
        self.use_h_loss = config.use_h_loss
        self.W_statistic = nn.Linear(2,config.d_model) 
        self.cls = nn.Parameter(torch.randn(1,config.d_model)*1e-2)
        
        self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.pred_len) 

        self.Attentions_over_token = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Attenion(config,over_hidden = True) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model,config.d_model)  for i in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout)  for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2)) for i in range(config.e_layers)])       

        self.time_of_day_size = config.time_of_day_size

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:

        z = x_enc.transpose(1,2)
        b,c,s = z.shape
        # use-norm
        z_mean = torch.mean(z,dim = (-1),keepdims = True)
        z_std = torch.std(z,dim = (-1),keepdims = True)
        z =  (z - z_mean)/(z_std + 1e-4)

        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                 
        z_embed = self.input_dropout(self.W_input_projection(zcube))+ self.W_pos_embed

        if self.use_statistic:
            z_stat = torch.cat((z_mean,z_std),dim = -1)
            if z_stat.shape[-2]>1:
                z_stat = (z_stat - torch.mean(z_stat,dim =-2,keepdims = True))/( torch.std(z_stat,dim =-2,keepdims = True)+1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2),z_embed),dim = -2)

        else:
            cls_token = self.cls.repeat(z_embed.shape[0],z_embed.shape[1],1,1)
            z_embed = torch.cat((cls_token,z_embed),dim = -2)

        inputs = z_embed
        b,c,t,h = inputs.shape 
        for a_2,a_1,mlp,drop,norm  in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output_1 = a_1(inputs.permute(0,2,1,3)).permute(0,2,1,3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            inputs = outputs
        
        # de-norm
        z_out = self.W_out(outputs.reshape(b,c,-1))  
        z = z_out *(z_std+1e-4)  + z_mean 

        return z

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        history_data[..., 1] = history_data[..., 1] * self.time_of_day_size // (self.time_of_day_size / 24) / 23.0
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
        future_data=future_data, start_token_len=0)
        #print(x_mark_enc.shape, x_mark_dec.shape)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)

        return {"prediction": prediction.transpose(1,2).unsqueeze(-1), 
                "use_h_loss": self.use_h_loss,
                "pred_len": self.pred_len}

