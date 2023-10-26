import torch
import torch.nn as nn


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.model_args = model_args

    def forward(self, X, H, st_l_conv, dynamic_graph, static_graph):
        [B, seq_len_remain, B, D] = H.shape
        [B, seq_len_input, B, D] = X.shape

        predict = []
        history = X
        predict.append(H[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(st_l_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict
