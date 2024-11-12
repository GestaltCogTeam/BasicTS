from .DynamicGCN import *
from .TCN import *

from argparse import Namespace
from basicts.utils import data_transformation_4_xformer

class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int,
                 M, dy_embedding_dim,
                 skip_channels: int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout: float, D: int, LowRank:int
                 ):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))

        self.s_conv = dynamicGCN(conv_channels, residual_channels, gcn_depth,dropout,propalpha,M,num_nodes,dy_embedding_dim)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))

        self.norm = LayerNorm((residual_channels, num_nodes, t_len), elementwise_affine=layer_norm_affline)
        #todo
        self.D = D
        self.Linear_query = nn.Linear(dy_embedding_dim * num_nodes+4, D)
        self.Linear_key = nn.Linear(LowRank,D)


    def weight_generation(self,x,sigma,x_mark_enc):
        B,C,N,T = x.shape
        x = x.transpose(1,3) #BTNC
        x = x.reshape(B,T,-1)

        x = torch.cat((x, x_mark_enc), dim=-1)

        query = self.Linear_query(x)
        key = self.Linear_key(sigma)
        score = torch.einsum('BTD,DK->BTK',query,key.transpose(1,0))/torch.sqrt(torch.tensor(self.D))
        score=F.softmax(score,dim=-1)

        return score

    def forward(self, x,x_mark_enc,U,V,sigma):


        residual = x

        x,x_mark_enc = self.t_conv(x,x_mark_enc)
        skip = self.skip_conv(x)
        weight = self.weight_generation(x,sigma,x_mark_enc)
        x = self.s_conv(x,U,V,weight,sigma)
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip, x_mark_enc


class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len: int, kernel_set, dilation_exp: int, n_layers: int,
                 residual_channels: int, conv_channels: int,
                 gcn_depth: int, M, dy_embedding_dim, skip_channels: int, num_nodes: int,
                 layer_norm_affline, propalpha: float, dropout: float,D,LowRank):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1 + block_id * (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        else:
            rf_block = block_id * n_layers * (kernel_size - 1) + 1

        dilation_factor = 1
        for i in range(1, n_layers + 1):
            if dilation_exp > 1:
                rf_size_i = int(rf_block + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = rf_block + i * (kernel_size - 1)
            t_len_i = total_t_len - rf_size_i + 1

            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, M,
                          dy_embedding_dim,
                          skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout,D,LowRank)
            )
            dilation_factor *= dilation_exp

    def forward(self, x ,x_mark_enc, U,V,skip_list,sigma):

        for layer in self:
            x, skip,x_mark_enc = layer(x,x_mark_enc, U,V, sigma)
            skip_list.append(skip)
        return x, skip_list


class Sumba(nn.Module):
    '''
    Paper: Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics
    Official Code: https://github.com/chenxiaodanhit/Sumba/
    Link: https://xiucheng.org/assets/pdfs/nips24-sumba.pdf
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    '''
    def __init__(self, **model_args):
        super(Sumba, self).__init__()
        config = Namespace(**model_args)
        self.n_blocks = 1
        self.gcn_true = config.gcn_true
        # self.buildA_true = config.buildA_true
        self.num_nodes = config.num_nodes
        self.dropout = config.dropout_MTGNN
        self.pred_len=config.pred_len
        # self.predefined_A = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.Linear_weight_T = nn.ModuleList()
        self.M = config.M
        self.LowRank = config.LowRank
        self.config = config
        self.start_conv = nn.Conv2d(in_channels=config.input_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1, 1))

        self.seq_length = config.seq_len
        kernel_size = 7
        if config.dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(config.dilation_exponential**config.layers-1)/(config.dilation_exponential-1))
        else:
            self.receptive_field = config.layers*(kernel_size-1) + 1


        self.total_t_len = max(self.receptive_field, self.seq_length)
        self.blocks = nn.ModuleList()

        for block_id in range(self.n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, config.kernel_set, config.dilation_exponential, config.sumba_layers, config.residual_channels, config.conv_channels, config.gcn_depth,
                 config.M, config.dimension, config.skip_channels, config.num_nodes, config.layer_norm_affline, config.propalpha, config.dropout_MTGNN,config.D,config.LowRank))



        for i in range(1):
            if config.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(config.dilation_exponential**config.layers-1)/(config.dilation_exponential-1))
            else:
                rf_size_i = i*config.layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,config.layers+1):
                if config.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(config.dilation_exponential**j-1)/(config.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(config.residual_channels, config.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(config.residual_channels, config.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                    self.Linear_weight_T.append(nn.Linear(self.seq_length-rf_size_j+1, config.dimension))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=config.conv_channels,
                                                    out_channels=config.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                    self.Linear_weight_T.append(nn.Linear(self.receptive_field-rf_size_j+1, config.dimension))
                if self.gcn_true:
                    self.gconv1.append(dynamicGCN(config.conv_channels, config.residual_channels, config.gcn_depth, config.dropout_MTGNN, config.propalpha,self.M,config.num_nodes,config.dimension))
                    self.gconv2.append(dynamicGCN(config.conv_channels, config.residual_channels, config.gcn_depth, config.dropout_MTGNN, config.propalpha,self.M,config.num_nodes,config.dimension))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((config.residual_channels,config. num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=True))
                else:
                    self.norm.append(LayerNorm((config.residual_channels, config.num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=True))

                new_dilation *= config.dilation_exponential

        self.layers = config.layers
        self.end_conv_1 = nn.Conv2d(in_channels=config.skip_channels,
                                             out_channels=config.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=config.end_channels,
                                             out_channels=config.output_dim*config.pred_len,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=config.input_dim, out_channels=config.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=config.residual_channels, out_channels=config.skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=config.input_dim, out_channels=config.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=config.residual_channels, out_channels=config.skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to('cuda:0')

        self.Sigma = nn.Parameter(torch.randn(self.M, self.LowRank), requires_grad=True)

        self.U = nn.Parameter(torch.randn(self.num_nodes, self.LowRank), requires_grad=True)
        self.V = nn.Parameter(torch.randn(self.num_nodes, self.LowRank),
                              requires_grad=True)

        in_channels = config.skip_channels
        final_channels = config.pred_len * config.output_dim
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, config.end_channels, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(config.end_channels, final_channels, kernel_size=(1, 1), bias=True)
        )

        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        self.time_of_day_size = config.time_of_day_size

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:
        
        """Feed forward of Sumba. Kindly note that only `x_enc`and `x_mark_enc` are actually used.
        
        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        input = x_enc
        means = input.mean(1, keepdim=True).detach()
        input = input - means
        stdev = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input /= stdev
        B, L, N = input.shape

        if(len(input.shape)==3):
            input=input.unsqueeze(-1)
            input=input.transpose(3,1)

        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))


        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        skip_list = [skip]

        for i in range(self.n_blocks):

            x, skip_list = self.blocks[i](x,x_mark_enc, self.U, self.V, skip_list, self.Sigma)

        skip_list.append(self.skipE(x))
        skip_list = torch.cat(skip_list, -1)

        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)
        x = self.out(skip_sum)
        output = x.reshape(B, self.pred_len, -1, N).transpose(-1, -2).squeeze()


        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return output

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        # change MinuteOfDay to MinuteOfHour
        history_data[..., 1] = history_data[..., 1] * self.time_of_day_size // (self.time_of_day_size / 24) / 23.0
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data, future_data=future_data, start_token_len=0)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction.unsqueeze(-1)