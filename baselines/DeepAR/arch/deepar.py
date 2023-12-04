import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Gaussian


class DeepAR(nn.Module):
    """
    Paper: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks; Link: https://arxiv.org/abs/1704.04110; Ref Code: https://github.com/jingw2/demand_forecast, https://github.com/husnejahan/DeepAR-pytorch, https://github.com/arrigonialberto86/deepar.
    """

    def __init__(self, cov_feat_size, embedding_size, hidden_size, num_layers, use_ts_id, id_feat_size=0, num_nodes=0) -> None:
        """Init DeepAR.

        Args:
            cov_feat_size (int): covariate feature size (e.g. time in day, day in week, etc.).
            embedding_size (int): output size of the input embedding layer.
            hidden_size (int): hidden size of the LSTM.
            num_layers (int): number of LSTM layers.
            use_ts_id (bool): whether to use time series id to construct spatial id embedding as additional features.
            id_feat_size (int, optional): size of the spatial id embedding. Defaults to 0.
            num_nodes (int, optional): number of nodes. Defaults to 0.
        """
        super().__init__()
        self.use_ts_id = use_ts_id
        # input embedding layer
        self.input_embed = nn.Linear(1, embedding_size)
        # spatial id embedding layer
        if use_ts_id:
            assert id_feat_size > 0, "id_feat_size must be greater than 0 if use_ts_id is True"
            assert num_nodes > 0, "num_nodes must be greater than 0 if use_ts_id is True"
            self.id_feat = nn.Parameter(torch.empty(num_nodes, id_feat_size))
            nn.init.xavier_uniform_(self.id_feat)
        else:
            id_feat_size = 0
        # the LSTM layer
        self.encoder = nn.LSTM(embedding_size+cov_feat_size+id_feat_size, hidden_size, num_layers, bias=True, batch_first=True)
        # the likelihood function
        self.likelihood_layer = Gaussian(hidden_size, 1)

    def gaussian_sample(self, mu, sigma):
        """Sampling.

        Args:
            mu (torch.Tensor): mean values of distributions.
            sigma (torch.Tensor): std values of distributions.
        """
        mu = mu.squeeze(1)
        sigma = sigma.squeeze(1)
        gaussian = torch.distributions.Normal(mu, sigma)
        ypred = gaussian.sample([1]).squeeze(0)
        return ypred

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of DeepAR.
        Reference code: https://github.com/jingw2/demand_forecast/blob/master/deepar.py

        Args:
            history_data (torch.Tensor): history data. [B, L, N, C].
            future_data (torch.Tensor): future data. [B, L, N, C].
            train (bool): is training or not.
        """
        history_next = None
        preds = []
        mus = []
        sigmas = []
        len_in, len_out = history_data.shape[1], future_data.shape[1]
        B, _, N, C = history_data.shape
        input_feat_full = torch.cat([history_data[:, :, :, 0:1], future_data[:, :, :, 0:1]], dim=1) # B, L_in+L_out, N, 1
        covar_feat_full = torch.cat([history_data[:, :, :, 1:], future_data[:, :, :, 1:]], dim=1) # B, L_in+L_out, N, C-1

        for t in range(1, len_in + len_out):
            if not (t > len_in and not train): # not in the decoding stage when inferecing
                history_next = input_feat_full[:, t-1:t, :, 0:1]
            else:
                a = 1
            embed_feat = self.input_embed(history_next)
            covar_feat = covar_feat_full[:, t:t+1, :, :]
            if self.use_ts_id:
                id_feat = self.id_feat.unsqueeze(0).expand(history_data.shape[0], -1, -1).unsqueeze(1)
                encoder_input = torch.cat([embed_feat, covar_feat, id_feat], dim=-1)
            else:
                encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
            # lstm
            B, _, N, C = encoder_input.shape # _ is 1
            encoder_input = encoder_input.transpose(1, 2).reshape(B * N, -1, C)
            _, (h, c) = self.encoder(encoder_input) if t == 1 else self.encoder(encoder_input, (h, c))
            # distribution proj
            mu, sigma = self.likelihood_layer(F.relu(h[-1, :, :]))
            history_next = self.gaussian_sample(mu, sigma).view(B, N).view(B, 1, N, 1)
            mus.append(mu.view(B, N, 1).unsqueeze(1))
            sigmas.append(sigma.view(B, N, 1).unsqueeze(1))
            preds.append(history_next)
            assert not torch.isnan(history_next).any()

        preds = torch.concat(preds, dim=1)
        mus = torch.concat(mus, dim=1)
        sigmas = torch.concat(sigmas, dim=1)
        reals = input_feat_full[:, -preds.shape[1]:, :, :]
        return preds, reals, mus, sigmas
