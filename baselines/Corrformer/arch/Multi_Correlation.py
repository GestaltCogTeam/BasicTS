import math

import torch
import torch.nn as nn
import torch.fft


class AutoCorrelation(nn.Module):
    # Auto-Correlation for temporal modeling of leaf nodes
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def values_delay_full(self, values, corr):
        # time delay agg (fully defined)
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        delay = length - delay
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # cal res
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def values_delay_channel(self, values, corr):
        # time delay agg (reduce channel)
        # b h c l
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(corr, dim=2)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # cal res
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[..., i].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # auto-correlation for temporal modeling
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        # values
        if self.training:
            V = self.values_delay_channel(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.values_delay_channel(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # B*N L H D
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class CrossCorrelation(nn.Module):
    # Cross-Correlation for spatial modeling of intermedia nodes
    def __init__(self, causal_fusion, mask_flag=True, factor=1, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super(CrossCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.causal_fusion = causal_fusion

    def align_speed(self, values, corr):
        # align (speed up)
        # B N H E L
        batch = values.shape[0]
        node = values.shape[1]
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[4]
        # index
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, node, head,
                                                                                                     channel, 1).cuda()
        # align
        rank = torch.topk(torch.mean(corr, dim=3), self.factor, dim=-1)
        weight = rank[0]
        delay = rank[1]
        delay = length - delay
        # cal res
        tmp_values = values.repeat(1, 1, 1, 1, 2)
        aligned_values = []
        for i in range(self.factor):
            tmp_delay = init_index + delay[..., i] \
                .unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, channel, 1)
            aligned_values.append(torch.gather(tmp_values, dim=-1, index=tmp_delay))
        return aligned_values, delay, weight

    def align_back_speed(self, values, delay):
        # align back (speed up)
        # B N H E L
        batch = values[0].shape[0]
        node = values[0].shape[1]
        head = values[0].shape[2]
        channel = values[0].shape[3]
        length = values[0].shape[4]
        # index
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, node, head,
                                                                                                     channel, 1).cuda()
        # algin back
        delay = length - delay
        # cal res
        agg_values = []
        for i in range(self.factor):
            tmp_values = values[i].repeat(1, 1, 1, 1, 2)
            tmp_delay = init_index + (delay[..., i]) \
                .unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, channel, 1)  # B N H
            aligned_values = torch.gather(tmp_values, dim=-1, index=tmp_delay)  # B N H E L
            agg_values.append(aligned_values)
        agg_values = torch.mean(torch.stack(agg_values, dim=0), dim=0)  # mean or sum
        return agg_values

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, H, E = queries.shape
        _, _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :, :(L - S), :, :]).float()
            values = torch.cat([values, zeros], dim=2)
            keys = torch.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :, :]
            keys = keys[:, :, :L, :, :]

        # cross-correlation for spatial modeling
        ## average pool for pivot series
        q_fft = torch.fft.rfft(
            torch.mean(queries, dim=1).unsqueeze(1).repeat(1, N, 1, 1, 1).permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        ## pivot-based cross correlation
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)  # B N H E L
        # align
        aligned_values, delay, weight = self.align_speed(values.permute(0, 1, 3, 4, 2).contiguous(), corr)
        # (weight B N H K, delay B N H K, aligned_values B N H E L K
        delay = delay.permute(0, 2, 1, 3).contiguous().view(B * H, N, self.factor)
        weight = weight.permute(0, 2, 1, 3).contiguous().view(B * H, N, self.factor)
        causal_agg_values = []
        for i in range(self.factor):
            # sort (weight B*H N, delay B*H N)
            _, indices = torch.sort(delay[..., i])
            sort_indices = indices[:, :, None, None].repeat(1, 1, E, L)
            # rearrange
            sorted_aligned_values = (
                    (aligned_values[i].permute(0, 2, 1, 3, 4).contiguous().view(B * H, N, E, L))
                    * torch.sigmoid(weight[..., i])[:, :, None, None].repeat(1, 1, E, L)) \
                .gather(dim=1, index=sort_indices)  # B*H N E L
            # aggregate
            sorted_aligned_values = sorted_aligned_values.view(B * H, N, E * L).permute(0, 2, 1).contiguous()
            sorted_aligned_values = self.causal_fusion(sorted_aligned_values)
            sorted_aligned_values = sorted_aligned_values.permute(0, 2, 1).contiguous().view(B * H, N, E, L)
            # sort back
            _, reverse_indices = torch.sort(indices)  # B*H N
            reverse_indices = reverse_indices[:, :, None, None].repeat(1, 1, E, L)  # B*H N E L
            # rearrange back
            sorted_aligned_values = sorted_aligned_values.gather(dim=1, index=reverse_indices)  # B*H N E L
            sorted_aligned_values = sorted_aligned_values.view(B, H, N, E, L).permute(0, 2, 1, 3, 4).contiguous()
            causal_agg_values.append(sorted_aligned_values)
        # align back
        delay = delay.view(B, H, N, self.factor).permute(0, 2, 1, 3).contiguous()
        V = self.align_back_speed(causal_agg_values, delay).permute(0, 1, 4, 2, 3)  # B N L H E

        if self.output_attention:
            return (V.contiguous(), corr)
        else:
            return (V.contiguous(), None)


class CrossCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(CrossCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_nodes):
        # B*N L D
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # B N L H D/H
        queries = self.query_projection(queries).view(B // n_nodes, n_nodes, L, H, -1)
        keys = self.key_projection(keys).view(B // n_nodes, n_nodes, S, H, -1)
        values = self.value_projection(values).view(B // n_nodes, n_nodes, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        # B, N, L, H, E
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class MultiCorrelation(nn.Module):
    # Unified the auto-correlation and cross-correlation on a multiscale tree structure
    def __init__(self, auto_correlation, cross_correlation, node_num, nodel_list, dropout=0.1):
        super(MultiCorrelation, self).__init__()
        self.node_num = node_num
        self.node_list = nodel_list
        self.auto_correlation = auto_correlation
        self.cross_correlation = cross_correlation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, mask=None):
        # Note: for self-version, cross=x
        # auto-correlation on leaf nodes
        x_leaf = self.dropout(self.auto_correlation(
            x, cross, cross,
            attn_mask=mask
        )[0])
        # st corr
        x_intermedia = []
        B, L, C = x.shape
        B, S, C = cross.shape
        # obtain the multiscale structure
        x_expand = x.view([B // self.node_num] + self.node_list + [L, C])  # B nodel1 nodel2 ... L C
        cross_expand = cross.view([B // self.node_num] + self.node_list + [S, C])  # B nodel1 nodel2 ... S C
        for i in range(len(self.node_list)):
            # prepare
            keep_node = 1
            for j in range(i):
                keep_node *= self.node_list[j]
            reduce_node = 1
            for j in range(i + 1, len(self.node_list)):
                reduce_node *= self.node_list[j]
            # keep node
            x_tmp = x_expand.view(
                [(B // self.node_num) * keep_node * self.node_list[i]] + self.node_list[(i + 1):] + [L, C])
            cross_tmp = cross_expand.view(
                [(B // self.node_num) * keep_node * self.node_list[i]] + self.node_list[(i + 1):] + [S, C])
            # reduce node for multi-scales
            for j in range(len(self.node_list) - i - 1):
                x_tmp = torch.mean(x_tmp, dim=1)
                cross_tmp = torch.mean(cross_tmp, dim=1)
            # x_tmp: B*keep_node*node_list[i], L, C; cross_tmp: B*keep_node*node_list[i], S, C
            # multi-correlation in a certain layer
            x_tmp = \
                self.cross_correlation(x_tmp, cross_tmp, cross_tmp, attn_mask=mask,
                                       n_nodes=self.node_list[i])[0] \
                    .unsqueeze(1).repeat(1, reduce_node, 1, 1)  # x_tmp: B*keep_node*node_list[i], reduce_node, L, C
            x_intermedia.append(x_tmp.view([B // self.node_num] + self.node_list + [L, C]))
        x_intermedia = torch.mean(torch.stack(x_intermedia, dim=0), dim=0).view(B, L, C)
        res = self.dropout(x_leaf) + self.dropout(x_intermedia)
        return res