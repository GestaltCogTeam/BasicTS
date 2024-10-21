import torch.optim as optim
from model import *
import metrics

class train_pipeline():
    def __init__(self, scaler, input_length, output_length, in_dim, num_nodes, nhid, dropout, lrate, wdecay, device):
        self.model = linear_transformer(input_length, output_length, in_dim, num_nodes, nhid, dropout)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output, _ = self.model(input)
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

    def eval(self, input, real_val):
        self.model.eval()
        output, _ = self.model(input)
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = metrics.masked_mape(predict,real,0.0).item()
        rmse = metrics.masked_rmse(predict,real,0.0).item()
        return loss.item(), mape, rmse
