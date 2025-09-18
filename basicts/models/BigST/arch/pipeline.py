import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import metrics
from bigst import bigst

class train_pipeline():
    def __init__(self, scaler, seq_num, in_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                 use_residual, use_bn, use_spatial, use_long, dropout, lrate, wdecay, device, supports, edge_indices):
        self.model = bigst(device, seq_num, in_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                           use_residual, use_bn, use_spatial, use_long, dropout, supports=supports, edge_indices=edge_indices)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.use_spatial = use_spatial
        self.clip = 5

    def train(self, input, real_val, feat=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_spatial:
            output, spatial_loss = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)-0.3*spatial_loss
        else:
            output, _ = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
        
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = metrics.masked_mape(predict,real,0.0).item()
        rmse = metrics.masked_rmse(predict,real,0.0).item()
        return loss.item(), mape, rmse
    
    def eval(self, input, real_val, feat=None, flag='overall'):
        if flag=='overall':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = metrics.masked_mape(predict,real,0.0).item()
            rmse = metrics.masked_rmse(predict,real,0.0).item()
            return loss.item(), mape, rmse
        elif flag=='horizon':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = self.scaler.inverse_transform(real_val)
            predict = self.scaler.inverse_transform(output)
            loss = []
            mape = []
            rmse = []
            for i in range(12):
                loss.append(self.loss(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())
            return loss, mape, rmse
