import torch
import torch.nn as nn


class SIN(nn.Module):
    def __init__(self, **model_args):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(SIN, self).__init__()
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.mlp1 = nn.Parameter(torch.ones(self.seq_len))
        self.mlp2 = nn.Parameter(torch.ones(self.seq_len, self.pred_len))
        self.state = nn.Linear(self.seq_len, 1)


    def forward(self, x, mode:str):
        x = x.transpose(1, 2)
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        x = x.transpose(1, 2)

        return x


    def _get_statistics(self, x):
        # print(x.shape, self.mlp1.shape)
        self.history_state = x * self.mlp1 * torch.tensor(0.05).to(x.device)
        self.pred_state = self.history_state @ self.mlp2 * torch.sqrt(torch.tensor(self.seq_len / self.pred_len).to(x.device)) * torch.tensor(0.05).to(x.device)

    def _normalize(self, x):
        x = x - self.history_state
        return x

    def _denormalize(self, x):
        x = x + self.pred_state
        return x