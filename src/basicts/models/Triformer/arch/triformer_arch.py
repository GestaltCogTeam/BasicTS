"""Triformer forecasting architecture."""

from basicts.modules.embed.time_features import calendar_features
from ..config.triformer_config import TriformerConfig
import math
import torch
import torch.nn as nn
from torch.nn import init


class Triformer(nn.Module):
    """
    Paper: Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2204.13767
    Ref Official Code: https://github.com/razvanc92/triformer
    Venue: IJCAI 2022
    Task: Long-term Time Series Forecasting
    """

    # The native Builder centralizes input normalization using these defaults.
    _input_normalization_defaults = {"eps": 1e-05, "affine": False}

    def __init__(self, config: TriformerConfig):
        super(Triformer, self).__init__()
        num_nodes = config.num_features
        lag = config.input_len
        horizon = config.output_len
        input_dim = config.input_dim
        channels = config.channels
        patch_sizes = config.patch_sizes
        mem_dim = config.memory_size
        self.factorized = True
        self.num_nodes = num_nodes
        self.num_nodes = num_nodes
        self.channels = channels
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)
        self.layers = nn.ModuleList()
        self.skip_generators = nn.ModuleList()
        self.horizon = horizon
        self.lag = lag
        cuts = lag
        for patch_size in patch_sizes:
            if cuts % patch_size != 0:
                raise Exception("Lag not divisible by patch size")
            cuts = int(cuts / patch_size)
            self.layers.append(
                Layer(
                    input_dim=channels,
                    num_nodes=num_nodes,
                    cuts=cuts,
                    cut_size=patch_size,
                    factorized=self.factorized,
                )
            )
            self.skip_generators.append(
                WeightGenerator(
                    in_dim=cuts * channels,
                    out_dim=256,
                    number_of_weights=1,
                    mem_dim=mem_dim,
                    num_nodes=num_nodes,
                    factorized=False,
                )
            )
        self.custom_linear = CustomLinear(factorized=False)
        self.projections = nn.Sequential(
            *[nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, horizon)]
        )
        self.output_len = config.output_len
        self.num_timestamps = getattr(config, "num_timestamps", 0)

    def get_parameter_number(self):
        total_num = sum((p.numel() for p in self.parameters()))
        trainable_num = sum((p.numel() for p in self.parameters() if p.requires_grad))
        return {"Total": total_num, "Trainable": trainable_num}

    def forward_xformer(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        x = self.start_fc(batch_x)
        batch_size = x.size(0)
        skip = 0
        for layer, skip_generator in zip(self.layers, self.skip_generators):
            x = layer(x)
            (weights, biases) = skip_generator()
            skip_inp = x.transpose(2, 1).reshape(batch_size, 1, self.num_nodes, -1)
            skip = skip + self.custom_linear(skip_inp, weights[-1], biases[-1])
        x = torch.relu(skip).squeeze(1)
        return self.projections(x).transpose(2, 1).unsqueeze(-1)

    def forward(self, inputs, inputs_timestamps=None):
        marks = calendar_features(
            inputs_timestamps,
            inputs,
            inputs.shape[1],
            self.start_fc.in_features - 1,
            center=False,
        )
        features = torch.cat(
            [
                inputs.unsqueeze(-1),
                marks.unsqueeze(2).expand(-1, -1, inputs.shape[2], -1),
            ],
            dim=-1,
        )
        return self.forward_xformer(features, None, None, None).squeeze(-1)


class Layer(nn.Module):
    def __init__(self, input_dim, num_nodes, cuts, cut_size, factorized):
        super(Layer, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.cuts = cuts
        self.cut_size = cut_size
        self.temporal_embeddings = nn.Parameter(
            torch.rand(cuts, 1, 1, self.num_nodes, 5), requires_grad=True
        )
        self.embeddings_generator = nn.ModuleList(
            [nn.Sequential(*[nn.Linear(5, input_dim)]) for _ in range(cuts)]
        )
        self.out_net1 = nn.Sequential(
            *[
                nn.Linear(input_dim, input_dim**2),
                nn.Tanh(),
                nn.Linear(input_dim**2, input_dim),
                nn.Tanh(),
            ]
        )
        self.out_net2 = nn.Sequential(
            *[
                nn.Linear(input_dim, input_dim**2),
                nn.Tanh(),
                nn.Linear(input_dim**2, input_dim),
                nn.Sigmoid(),
            ]
        )
        self.temporal_att = TemporalAttention(input_dim, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(
            input_dim,
            input_dim,
            mem_dim=5,
            num_nodes=num_nodes,
            factorized=factorized,
            number_of_weights=2,
        )
        self.weights_generator_shared = WeightGenerator(
            input_dim,
            input_dim,
            mem_dim=None,
            num_nodes=num_nodes,
            factorized=False,
            number_of_weights=2,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        self.temporal_embeddings = self.temporal_embeddings.to(x.device)
        batch_size = x.size(0)
        data_concat = None
        out = 0
        (weights_shared, biases_shared) = self.weights_generator_shared()
        (weights_distinct, biases_distinct) = self.weights_generator_distinct()
        for i in range(self.cuts):
            t = x[:, i * self.cut_size : (i + 1) * self.cut_size, :, :]
            if i != 0:
                out = self.out_net1(out) * self.out_net2(out)
            emb = (
                self.embeddings_generator[i](self.temporal_embeddings[i]).repeat(
                    batch_size, 1, 1, 1
                )
                + out
            )
            t = torch.cat([emb, t], dim=1)
            out = self.temporal_att(
                t[:, :1, :, :],
                t,
                t,
                weights_distinct,
                biases_distinct,
                weights_shared,
                biases_shared,
            )
            if data_concat == None:
                data_concat = out
            else:
                data_concat = torch.cat([data_concat, out], dim=1)
        return self.dropout(data_concat)


class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class TemporalAttention(nn.Module):
    def __init__(self, in_dim, factorized):
        super(TemporalAttention, self).__init__()
        self.K = 8
        if in_dim % self.K != 0:
            raise Exception(
                "Hidden size is not divisible by the number of attention heads"
            )
        self.head_size = int(in_dim // self.K)
        self.custom_linear = CustomLinear(factorized)

    def forward(
        self,
        query,
        key,
        value,
        weights_distinct,
        biases_distinct,
        weights_shared,
        biases_shared,
    ):
        batch_size = query.shape[0]
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))
        attention = torch.matmul(query, key)
        attention /= self.head_size**0.5
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.tanh(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x


class WeightGenerator(nn.Module):
    def __init__(
        self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4
    ):
        super(WeightGenerator, self).__init__()
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(
                torch.randn(num_nodes, mem_dim), requires_grad=True
            )
            self.generator = self.generator = nn.Sequential(
                *[
                    nn.Linear(mem_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 5 * 5),
                ]
            )
            self.P = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
            self.Q = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.Tensor(self.mem_dim, out_dim), requires_grad=True
                    )
                    for _ in range(number_of_weights)
                ]
            )
            self.B = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.Tensor(self.mem_dim**2, out_dim), requires_grad=True
                    )
                    for _ in range(number_of_weights)
                ]
            )
        else:
            self.P = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
            self.B = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))
        if not self.factorized:
            for i in range(self.number_of_weights):
                (fan_in, _) = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [
                torch.matmul(memory, self.B[i]).squeeze(1)
                for i in range(self.number_of_weights)
            ]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [
                torch.matmul(torch.matmul(self.P[i], memory), self.Q[i])
                for i in range(self.number_of_weights)
            ]
            return (weights, bias)
        else:
            return (self.P, self.B)
