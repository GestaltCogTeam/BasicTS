import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None ):
        super(conv2d_, self).__init__()
        self.activation = activation
        # self.dropout = dropout
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        # (batch_size, num_step, num_vertex, D)
        x = x.permute(0, 3, 2, 1) # (batch_size , D, num_step, num_vertex)

        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))

        x = self.conv(x)

        x = self.batch_norm(x)

        if self.activation is not None :
            x = self.activation(x)

        return x.permute(0, 3, 2, 1)

class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list 
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        # print("before x:", x.shape)
        for conv in self.convs:
            x = conv(x)
        # print("after x: ", x.shape)
        return x
    


class TimeEncode(nn.Module):
    '''
    X: (batch_size, num_his+num_pred, 2)(dayofweek, timeofday)
    '''
    def __init__(self,D,bn_decay):
        super(TimeEncode,self).__init__()
        # Y = Wx+b
        self.ff = nn.Linear(2,2)
        self.FC_his = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, F.sigmoid],
            bn_decay=bn_decay)
        self.FC_Pred = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)
    def forward(self, x, num_vertex, num_his):
        x = x.float()
        x = self.ff(x)
        x = torch.sin(x)
        x = x.unsqueeze(dim = 2)
        His = x[:, :num_his]
        Pred = x[:, num_his:]
        His = self.FC_his(His)
        Pred = self.FC_Pred(Pred)
        add_vertex = torch.zeros(1,1,num_vertex,1)
        add_vertex = add_vertex #.to(device)
        His = His + add_vertex
        Pred = Pred + add_vertex
        del x, add_vertex
        return His, Pred
    
class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x
    
class TEmbedding(nn.Module):
    '''
    X: (batch_size, num_his+num_pred, 2) (dayofweek, timeofday)
    T: num of time steps in one day
    return: (batch_size, num_his+num_pred, num_vertex, D)
    '''
    def __init__(self, input_dim, D, num_nodes,bn_decay,) -> None:
        super(TEmbedding, self).__init__()
        self.FC = FC(input_dims=[input_dim, D, D], units=[D, D, D], activations=[F.relu, F.relu, F.sigmoid],
            bn_decay=bn_decay)
        
    def forward(self, X, SE, T, num_vertex, num_his):
        dayofweek = torch.empty(X.shape[0], X.shape[1], 7)
        timeofday = torch.empty(X.shape[0], X.shape[1], T)
        for i in range(X.shape[0]): # shape[0] = batch_size
            dayofweek[i] = F.one_hot(X[..., 0][i].to(torch.int64) % 7, 7) 
        for j in range(X.shape[0]):
            timeofday[j] = F.one_hot(X[..., 1][j].to(torch.int64) % T, T)
        X_in = torch.cat((timeofday, dayofweek), dim=-1) # (batch_size, num_his+num_pred, 7+T)
        X_in = X_in.unsqueeze(dim = 2)
        X_in = X_in.to(X.device) # (batch_size, num_his+num_pred, 1, D)
        add_vertex = torch.zeros(1,1,num_vertex,1)
        add_vertex = add_vertex.to(X.device)
        X_in = X_in + add_vertex
        X_in = self.FC(X_in)
        X_in = torch.sin(X_in)
        His = X_in[:, :num_his]
        Pred = X_in[:, num_his:]
        del dayofweek, timeofday, add_vertex, X_in
        return His + F.relu(SE), Pred

class SEmbedding(nn.Module):
    def __init__(self, D):
        super(SEmbedding, self).__init__()

        self.LaplacianPE1 = nn.Linear(32, 32)
        self.Norm1 = nn.LayerNorm(32, elementwise_affine=False)
        self.act = nn.LeakyReLU()
        self.LaplacianPE2 = nn.Linear(32, D)
        self.Norm2 = nn.LayerNorm(D, elementwise_affine=False)

    def forward(self, lpls, batch_size, pred_steps):
        lap_pos_enc = self.Norm2(self.LaplacianPE2(self.act(self.Norm1(self.LaplacianPE1(lpls)))))
        tensor_neb = lap_pos_enc.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, pred_steps, 1, 1)
        return F.sigmoid(tensor_neb)

class Trend(nn.Module):
    '''
    X: (batch_size, num_step, num_vertex, D)
    TEmbedding： (batch_size, num_step, num_vertex, D)
    return: (batch_size, num_step, num_vertex, D)
    '''
    def __init__(self):
        super(Trend, self).__init__()
        
    def forward(self, X, STEmbedding): 
        return torch.mul(X, STEmbedding)

class Seasonal(nn.Module):
    '''
    X: (batch_size, num_step, num_vertex, D)
    return: (batch_size, num_step, num_vertex, D)
    '''
    def __init__(self):
        super(Seasonal, self).__init__()

    def forward(self, X, Trend):
        return X-Trend

class Trend_Seasonal_Decomposition(nn.Module):
    def __init__(self, num_vertex):
        super(Trend_Seasonal_Decomposition, self).__init__()
        self.vector = nn.Parameter(torch.full((1, 1, num_vertex, 1), 0.5, requires_grad=True))
    def forward(self, X, STEmbedding):
        trend = torch.mul(X, STEmbedding)
        seasonal = X-trend
        zero_shape=torch.zeros_like(X).cuda()
        vector = zero_shape+self.vector
        result = vector*trend+(1-vector)*seasonal
        del trend, seasonal
        return result

def adjacency_to_edge_index(adj):
    edge_index = adj.nonzero(as_tuple=False).t()
    return edge_index

class GCN(torch.nn.Module):
    def __init__(self, num_features, out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, out)

    def forward(self, data,batch_size, pred_steps):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        x = x.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, pred_steps, 1, 1)
        return x

        
class MAB(nn.Module):
    def __init__(self, K,d,input_dim,output_dim,bn_decay):
        super(MAB, self).__init__()
        D=K*d
        self.K = K
        self.d=d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, Q, K,batch_size,type="spatial",mask=None):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        if mask==None:
            if type=="temporal":
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
            attention = torch.matmul(query, key.transpose(2, 3))
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
            result = torch.matmul(attention, value)
            if type=="temporal":
                result = result.permute(0, 2, 1, 3)
            result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
            result = self.FC(result)
        else:
            mask=torch.cat(torch.split(mask, self.K, dim=-1), dim=0)
            if type=="temporal":
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
                mask=mask.permute(0,2,1,3)
            if mask.shape==query.shape:
                set_mask=torch.ones_like(key).cuda()
                mask = torch.matmul(mask,set_mask.transpose(2,3))
            elif mask.shape==key.shape:
                set_mask=torch.ones_like(query).cuda()
                mask = torch.matmul(set_mask,mask.transpose(2,3))
            attention = torch.matmul(query, key.transpose(2, 3))
            attention /= (self.d ** 0.5)
            attention=attention.masked_fill(mask==0,-1e9)
            attention = F.softmax(attention, dim=-1)
            result = torch.matmul(attention, value)
            if type=="temporal":
                result = result.permute(0, 2, 1, 3)
            result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
            result = self.FC(result)
        return result
class MAB_new(nn.Module):
    def __init__(self, K,d,input_dim,output_dim,bn_decay):
        super(MAB_new, self).__init__()
        D=K*d
        self.K = K
        self.d=d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu,
                     bn_decay=bn_decay)
    def forward(self, Q, K,batch_size):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        result = torch.matmul(attention, value)
        result = result.permute(0, 2, 1, 3)
        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        result = self.FC(result)
        return result

class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, K, d,num_of_vertices,set_dim, bn_decay):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices=num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1,set_dim,self.num_of_vertices, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D, bn_decay)
        self.mab1 = MAB(K, d, D, D, bn_decay)
    def forward(self, X, mask):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size,"temporal",mask)
        result = self.mab1(X, H, batch_size,"temporal",mask)
        return torch.add(X, result)

class AttentionDecoder(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    Y:      [batch_size, num_step, num_vertex, D]
    TE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, K, d,num_of_vertices,set_dim, bn_decay):
        super(AttentionDecoder, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices=num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1,set_dim,self.num_of_vertices, 3*D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB_new(K, d, 3*D, 3*D, bn_decay)
        self.mab1 = MAB_new(K, d, 3*D, D, bn_decay)

    def forward(self, X, TE, SE, mask):
    # def forward(self, X, SE, mask):
        batch_size = X.shape[0]
        mid = X 
        X = torch.cat((X, TE, SE), dim=-1)
        # X = torch.add(X, TE)
        # [batch_size, num_step, num_vertex, K * d]
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size)
        result = self.mab1(X, H, batch_size)
        return torch.add(mid, result)
class GRU(nn.Module):
    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2*outfea, 2*outfea)
        self.zff = nn.Linear(2*outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r*xh], -1)))
        x = u * z + (1-u) * xh
        return x

class GRUEncoder(nn.Module):
    def __init__(self, outfea, num_step):
        super(GRUEncoder, self).__init__()
        self.gru = nn.ModuleList([GRU(outfea) for i in range(num_step)])
        
    def forward(self,x):
        B,T,N,F = x.shape
        hidden_state = torch.zeros([B,N,F]).to(x.device)
        output = []
        for i in range(T):
            gx = x[:,i,:,:]
            gh = hidden_state
            hidden_state = self.gru[i](gx, gh)
            output.append(hidden_state)
        output = torch.stack(output, 1)
        return output

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()

# GCN
class gcn(nn.Module):
    """
    x:          [batch_size, num_step, num_vertex, D]
    support:    [num_vertex, D, D]
    """
    def __init__(self, c_in, c_out, dropout = 0.3, support_len = 1, order = 2, bn_decay = 0.1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len+1) * c_in
        self.mlp = FC(c_in, c_out, activations=F.relu, bn_decay=bn_decay)
        self.dropout = dropout
        self.order = order
    def forward(self, x, support):
        x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2            
        h = torch.cat(out, dim = 1)
        h = h.transpose(1,3)
        h = self.mlp(h)
        h = h.transpose(1,3)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1,3)
        return h

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # Initialize an empty list to store the results
#         outputs = []
#         for i in range(x.size(2)):  # Loop over number_nodes
#             node_data = x[:, :, i, :]  # Extract data for each node (batch_size, time_steps, channels)
#             front = node_data[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#             end = node_data[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#             node_data = torch.cat([front, node_data, end], dim=1)
#             node_data = self.avg(node_data.permute(0, 2, 1))  # (batch_size, channels, time_steps)
#             node_data = node_data.permute(0, 2, 1)  # Back to (batch_size, time_steps, channels)
#             outputs.append(node_data.unsqueeze(2))  # Add back the number_nodes dimension
        
#         # Concatenate along the number_nodes dimension
#         output = torch.cat(outputs, dim=2)
#         return output

# class series_decomp(nn.Module):
#     """
#     Series decomposition block for 4D traffic data
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean

class STDN(nn.Module):
    def __init__(self, args, bn_decay) -> None:
        super(STDN, self).__init__()
        data_config = args['Data']
        training_config = args['Training']
        L = int(training_config['L'])
        K = int(training_config['K'])
        d = int(training_config['d'])
        self.L=L
        self.K=K
        self.d=d
        self.node_miss_rate=float(training_config['node_miss_rate'])
        self.T_miss_len=int(training_config['T_miss_len'])
        self.order = int(training_config['order'])
        print('L',self.L)
        print('K',self.K)
        print('d',self.d)
        print('node_miss_rate',self.node_miss_rate)
        print('T_miss_len',self.T_miss_len)
        D = K * d
        set_dim = int(training_config['reference'])
        self.num_his = int(training_config['num_his'])
        time_slice_size = int(data_config['time_slice_size'])
        self.input_dim = int(1440/time_slice_size)+7
        self.num_pred=int(training_config['num_pred'])
        self.num_of_vertices=int(data_config['num_of_vertices'])
        self.TEmbedding = TEmbedding(self.input_dim , D, self.num_of_vertices,bn_decay)
        self.SEmbedding = SEmbedding(D)
        self.Trend = Trend()
        self.Seasonal = Seasonal()
        self.FeedForward_for_t = FeedForward([D,D], res_ln=True)
        self.FeedForward_for_s = FeedForward([D,D], res_ln=True)
        self.GRU_Trend = GRUEncoder(D, self.num_his)
        self.GRU_Seasonal = GRUEncoder(D, self.num_his)
        self.Decoder = nn.ModuleList([AttentionDecoder(K, d, self.num_of_vertices, set_dim, bn_decay) for _ in range(L)])
        self.dataset = data_config['dataset_name']
        in_channels = int(training_config['in_channels'])
        out_channels = int(training_config['out_channels'])

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)  # in_channels=3
        self.FC_2 = FC(input_dims=[D, D], units=[D,out_channels], activations=[F.relu, None],
                       bn_decay=bn_decay)

        # dynamic GCN
        self.nodevec_p1 = nn.Parameter(torch.randn(int(1440/time_slice_size), D), requires_grad=True) #.to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(int(data_config['num_of_vertices']), D), requires_grad=True) #.to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(int(data_config['num_of_vertices']), D), requires_grad=True) #.to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(D, D, D), requires_grad=True) #.to(device)
        self.GCN = gcn(D,D,order = self.order)
        
    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        # print(adp.shape)
        return adp   
        
    def forward(self, X, TE, lpls, mode, mask=None):
        # input
        X = self.FC_1(X)
        ind = TE[:,0,1]
        ind = torch.tensor(ind,dtype=torch.long)
        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        new_supports = [adp]
        X  = self.GCN(X, new_supports)

        SE = self.SEmbedding(lpls, X.shape[0] , self.num_pred)
        his, pred = self.TEmbedding(TE, SE, self.input_dim - 7, self.num_of_vertices, self.num_his)
        trend = self.Trend(X,his)
        seasonal = self.Seasonal(X,trend)
        trend = self.FeedForward_for_t(trend)
        seasonal = self.FeedForward_for_s(seasonal)
        # encoder
        trend = self.GRU_Trend(trend)
        seasonal = self.GRU_Seasonal(seasonal)

        result = trend + seasonal
        # decoder
        for net in self.Decoder:
            result = net(result, pred, SE, None)
        result = self.FC_2(result)
        del TE, his, trend, seasonal
        return result

def make_model( config, bn_decay=0.1):
    model = STDN( config, bn_decay=0.1)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model