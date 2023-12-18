import math

import torch
import torch.nn as nn


class GeoPositionalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GeoPositionalEmbedding, self).__init__()
        # Compute the national position
        self.location_embedding = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.location_embedding(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.day_of_month_size = day_of_month_size
        self.day_of_year_size = day_of_year_size

        if time_of_day_size is not None:
            self.time_of_day_embed = Embed(time_of_day_size, d_model)
        if day_of_week_size is not None:
            self.day_of_week_embed = Embed(day_of_week_size, d_model)
        if day_of_month_size is not None:
            self.day_of_month_embed = Embed(day_of_month_size, d_model)
        if day_of_year_size is not None:
            self.day_of_year_embed = Embed(day_of_year_size, d_model)

    def forward(self, x):
        x = x + 0.5 # re-normalize to [0, 1) to get correct index. ref: arch_zoo/utils/xformer.py data_transformation_4_xformer
        temporal_embeddings = []
        if self.time_of_day_size is not None:
            time_of_day_x = self.time_of_day_embed((x[:, :, 0] * self.time_of_day_size).long())
            temporal_embeddings.append(time_of_day_x)
        if self.day_of_week_size is not None:
            day_of_week_x = self.day_of_week_embed((x[:, :, 1] * self.day_of_week_size).long())
            temporal_embeddings.append(day_of_week_x)
        if self.day_of_month_size is not None:
            day_of_month_x = self.day_of_month_embed((x[:, :, 2] * self.day_of_month_size).long())
            temporal_embeddings.append(day_of_month_x)
        if self.day_of_year_size is not None:
            day_of_year_x = self.day_of_year_embed((x[:, :, 3] * self.day_of_year_size).long())
            temporal_embeddings.append(day_of_year_x)

        if len(temporal_embeddings) == 0:
            return 0
        else:
            return sum(temporal_embeddings)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, num_time_features):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(num_time_features, d_model, bias=True)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, node_num, variable_num, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size, embed_type='fixed', num_time_features=-1, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.node_num = node_num
        self.national_embedding = GeoPositionalEmbedding(3 * (variable_num // node_num), d_model)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
                                                    time_of_day_size=time_of_day_size,
                                                    day_of_week_size=day_of_week_size,
                                                    day_of_month_size=day_of_month_size,
                                                     day_of_year_size=day_of_year_size,
                                                     embed_type=embed_type) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, num_time_features=num_time_features)

        self.dropout = nn.Dropout(p=dropout)
        self.national_position = nn.Parameter(torch.empty(node_num, 3 * (variable_num // node_num)))
        nn.init.xavier_uniform_(self.national_position)


    def forward(self, x, x_mark):
        B, L, D = x.shape
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + self.national_embedding(self.national_position) \
                .unsqueeze(1).unsqueeze(0).repeat(B // self.node_num, 1, L, 1) \
                .view(B, L, -1)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size, embed_type='fixed', num_time_features=-1, dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
                                                    time_of_day_size=time_of_day_size,
                                                    day_of_week_size=day_of_week_size,
                                                    day_of_month_size=day_of_month_size,
                                                     day_of_year_size=day_of_year_size,
                                                     embed_type=embed_type) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, num_time_features=num_time_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        value_embed = self.value_embedding(x)
        temp_embed = self.temporal_embedding(x_mark)
        x = value_embed + temp_embed

        return self.dropout(x)
