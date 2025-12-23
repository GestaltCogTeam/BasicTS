# pylint: disable=not-callable
import torch
import torch.nn.functional as F
from torch import nn

from ..config.stemgnn_config import StemGNNConfig


class GLU(nn.Module):
    """
    Gated Linear Unit
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear_left = nn.Linear(input_size, output_size)
        self.linear_right = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))

class StockBlock(nn.Module):
    """
    StemGNN Block
    """
    def __init__(self,
                 input_len: int,
                 num_features: int,
                 hidden_size: int,
                 layer_idx: int):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.input_len * self.hidden_size,
                         self.hidden_size * self.input_len))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(
            self.input_len * self.hidden_size, self.input_len * self.hidden_size)
        self.forecast_result = nn.Linear(
            self.input_len * self.hidden_size, self.input_len)
        if self.layer_idx == 0:
            self.backcast = nn.Linear(
                self.input_len * self.hidden_size, self.input_len)
        self.backcast_short_cut = nn.Linear(self.input_len, self.input_len)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_hidden_size = 4 * self.hidden_size
        for i in range(3):
            if i == 0:
                self.GLUs.append(
                    GLU(self.input_len * 4, self.input_len * self.output_hidden_size))
                self.GLUs.append(
                    GLU(self.input_len * 4, self.input_len * self.output_hidden_size))
            elif i == 1:
                self.GLUs.append(
                    GLU(self.input_len * self.output_hidden_size, self.input_len * self.output_hidden_size))
                self.GLUs.append(
                    GLU(self.input_len * self.output_hidden_size, self.input_len * self.output_hidden_size))
            else:
                self.GLUs.append(
                    GLU(self.input_len * self.output_hidden_size, self.input_len * self.output_hidden_size))
                self.GLUs.append(
                    GLU(self.input_len * self.output_hidden_size, self.input_len * self.output_hidden_size))

    def spe_seq_cell(self, inputs: torch.Tensor) -> torch.Tensor:
        # Spectral Sequential Cell
        batch_size, _, _, num_features, input_len = inputs.size()
        inputs = inputs.view(batch_size, -1, num_features, input_len)
        ffted = torch.fft.fft(inputs, dim=-1)
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        ffted = torch.stack([ffted_real, ffted_imag], dim=-1)
        real = ffted[..., 0].permute(
            0, 2, 1, 3).contiguous().reshape(batch_size, num_features, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(
            batch_size, num_features, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, num_features, 4, -
                            1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, num_features, 4, -
                          1).permute(0, 2, 1, 3).contiguous()
        input_len_as_inner = torch.complex(real, img)
        iffted = torch.fft.ifft(input_len_as_inner, dim=-1).real
        return iffted

    def forward(
            self, x: torch.Tensor, graph: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = graph.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(graph, x)      # B, cheb_order, 1, N, L
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.layer_idx == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(
                self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class StemGNN(nn.Module):
    """
    Paper: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting
    Link: https://arxiv.org/abs/2103.07719
    Official Code: https://github.com/microsoft/StemGNN
    Venue: NeurIPS 2020
    Task: Spatial-Temporal Forecasting
    Note:  
        There are some difference in implementation described in the paper as well as the source code. 
        Details can be found in [here](https://github.com/microsoft/StemGNN/issues/12)
        We adopt the implementation of the code.
    Details of difference:
        - No reconstruction loss.
        - No 1DConv.
        - Use chebyshev polynomials to reduce time complexity.
        - There is no the output layer composed of GLU and fully-connected (FC) sublayers as described in third paragraph in section 4.1.
        - The experimental setting is not fair in StemGNN, and we can not reproduce the paper's performance.
    """

    def __init__(self, config: StemGNNConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.num_blocks = config.num_blocks
        self.hidden_size = config.hidden_size
        self.alpha = config.leaky_rate

        # Spectral Graph Convolution Parameters
        self.weight_key = nn.Parameter(torch.zeros(size=(self.num_features, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.num_features, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.input_len, self.num_features)

        # Backbone
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlock(self.input_len, self.num_features, self.hidden_size, i)
             for i in range(self.num_blocks)])
        # Forecasting layer
        self.fc = nn.Sequential(
            nn.Linear(int(self.input_len), int(self.input_len)),
            nn.LeakyReLU(),
            nn.Linear(int(self.input_len), self.output_len),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(config.dropout)

    def get_laplacian(self, graph: torch.Tensor, normalize: bool) -> torch.Tensor:
        # Laplacian matrix
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device,
                          dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian: torch.Tensor) -> torch.Tensor:
        # Graph Chebyshev polynomials
        num_features = laplacian.size(0)  # [num_features, num_features]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, num_features, num_features], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * \
            torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral Graph
        inputs, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        inputs = inputs.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(inputs)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        graph = self.cheb_polynomial(laplacian)
        return graph

    def self_graph_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        # Graph attention
        inputs = inputs.permute(0, 2, 1).contiguous()
        batch_size, num_features, _ = inputs.size()
        key = torch.matmul(inputs, self.weight_key)
        query = torch.matmul(inputs, self.weight_query)
        data = key.repeat(1, 1, num_features).view(
            batch_size, num_features * num_features, 1) + query.repeat(1, num_features, 1)
        data = data.squeeze(2)
        data = data.view(batch_size, num_features, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, inputs, eigenvectors):
        return torch.matmul(eigenvectors, inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Feedforward function of StemGNN.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """
        graph = self.latent_correlation_layer(inputs)
        x = inputs.unsqueeze(1).transpose(-1, -2).contiguous()
        result = []
        for stack_i in range(self.num_blocks):
            prediction, x = self.stock_block[stack_i](x, graph)
            result.append(prediction)
        prediction = result[0] + result[1]
        prediction = self.fc(prediction).transpose(1, 2)
        return prediction
