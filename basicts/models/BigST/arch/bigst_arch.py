import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_conv import *
from torch.autograd import Variable
import pdb
from .preprocess import BigSTPreprocess
from .model import Model

def sample_period(x, time_num):
    # trainx (B, N, T, F)
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i+12] for i in range(0, history_length, time_num)]
    period_feat = [x[:,:,sublist,0] for sublist in period_list]
    period_feat = torch.stack(period_feat)
    period_feat = torch.mean(period_feat, dim=0)
    
    return period_feat

class BigST(nn.Module):
    """
    Paper: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
    Link: https://dl.acm.org/doi/10.14778/3641204.3641217
    Official Code: https://github.com/usail-hkust/BigST?tab=readme-ov-file
    Venue: VLDB 2024
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, bigst_args, preprocess_path, preprocess_args):
        super(BigST, self).__init__()

        self.use_long = bigst_args['use_long']
        self.in_dim = bigst_args['in_dim']
        self.out_dim = bigst_args['out_dim']
        self.time_num = bigst_args['time_of_day_size']
        self.bigst = Model(**bigst_args) 

        if self.use_long:
            self.feat_extractor = BigSTPreprocess(**preprocess_args)
            self.load_pre_trained_model(preprocess_path)
            
    def load_pre_trained_model(self, preprocess_path):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(preprocess_path)
        self.feat_extractor.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.feat_extractor.parameters():
            param.requires_grad = False

        self.feat_extractor.eval()


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        history_data = history_data.transpose(1,2) # (B, N, T, D)
        x = history_data[:, :, -self.out_dim:]         # (batch_size, in_len, data_dim)

        if self.use_long:
            feat = []
            for i in range(history_data.shape[0]):
                with torch.no_grad():
                     feat_sample = self.feat_extractor(history_data[[i],:,:,:], future_data, batch_seen, epoch, train)
                feat.append(feat_sample['feat'])

            feat = torch.cat(feat, dim=0)
            feat_period = sample_period(history_data, self.time_num)
            feat = torch.cat([feat, feat_period], dim=2)

            return self.bigst(x, feat)

        else:
            return self.bigst(x)

        