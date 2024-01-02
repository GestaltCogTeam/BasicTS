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

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, train: bool, history_mask: torch.Tensor, future_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of DeepAR.
        Reference code: https://github.com/jingw2/demand_forecast/blob/master/deepar.py

        Args:
            history_data (torch.Tensor): history data. [B, L, N, C].
            future_data (torch.Tensor): future data. [B, L, N, C].
            train (bool): is training or not.
        """
        mask = torch.cat([history_mask, future_mask], dim=1).unsqueeze(-1)[:, 1:, ...]
        # mask = torch.where(mask == 0, torch.ones_like(mask) * 1e-5, mask)
        # mask = torch.ones_like(mask)
        # nornalization
        means = history_data.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(history_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        history_data_normed = history_data - means
        history_data_normed /= stdev
        future_data_normed = future_data - means
        future_data_normed /= stdev
        
        history_next = None
        preds = []
        mus = []
        sigmas = []
        len_in, len_out = history_data.shape[1], future_data.shape[1]
        B, _, N, C = history_data.shape
        input_feat_full_normed = torch.cat([history_data_normed[:, :, :, 0:1], future_data_normed[:, :, :, 0:1]], dim=1) # B, L_in+L_out, N, 1
        input_feat_full = torch.cat([history_data[:, :, :, 0:1], future_data[:, :, :, 0:1]], dim=1) # B, L_in+L_out, N, 1

        for t in range(1, len_in + len_out):
            if not (t > len_in and not train): # not in the decoding stage when inferecing
                history_next = input_feat_full_normed[:, t-1:t, :, 0:1]
            embed_feat = self.input_embed(history_next)
            # 检查nan
            assert not torch.isnan(history_next).any(), "history_next中存在nan"
            assert not torch.isnan(self.input_embed.weight).any(), "embed_feat中存在nan"
            assert not torch.isnan(self.input_embed.bias).any(), "embed_feat中存在nan"
            assert not torch.isnan(embed_feat).any(), "embed_feat中存在nan"
            encoder_input = embed_feat
            # lstm
            B, _, N, C = encoder_input.shape # _ is 1
            encoder_input = encoder_input.transpose(1, 2).reshape(B * N, -1, C)
            _, (h, c) = self.encoder(encoder_input) if t == 1 else self.encoder(encoder_input, (h, c))
            # distribution proj
            mu, sigma = self.likelihood_layer(h[-1, :, :])
            history_next = self.gaussian_sample(mu, sigma).view(B, N).view(B, 1, N, 1)
            mus.append(mu.view(B, N, 1).unsqueeze(1))
            sigmas.append(sigma.view(B, N, 1).unsqueeze(1))
            preds.append(history_next)
            assert not torch.isnan(history_next).any()

        preds = torch.concat(preds, dim=1)
        mus = torch.concat(mus, dim=1)
        sigmas = torch.concat(sigmas, dim=1)
        reals = input_feat_full[:, -preds.shape[1]:, :, :]
        
        # 检查mus和sigmas中是否存在nan
        assert not torch.isnan(mus).any(), "mus中存在nan"
        assert not torch.isnan(sigmas).any(), "sigmas中存在nan"

        # denormalization
        preds = preds * stdev + means
        mus = mus * stdev + means
        sigmas = sigmas * stdev + means

        return {"prediction": preds * mask, "target": reals * mask, "mus": mus, "sigmas": sigmas, "mask_prior": mask}
