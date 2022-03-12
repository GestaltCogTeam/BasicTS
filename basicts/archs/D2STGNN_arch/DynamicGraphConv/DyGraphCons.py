import torch.nn as nn
from basicts.archs.D2STGNN_arch.DynamicGraphConv.Utils import *

class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # model args
        self.k_s = model_args['k_s']  # spatial order
        self.k_t = model_args['k_t']  # temporal kernel size
        self.hidden_dim = model_args['num_hidden']              # hidden dimension of 
        self.node_dim   = model_args['node_hidden']         # trainable node embedding dimension

        self.distance_function  = DistanceFunction(hidden_dim=self.hidden_dim, node_dim=self.node_dim, **model_args)
        self.mask               = Mask(method='geograph', order=self.k_s, **model_args)
        self.normalizer         = Normalizer(method='transition', **model_args)         # TODO: 测试不要transition（因为mask已经有了）
        self.multi_order        = MultiOrder(method='multiplication', order=self.k_s, **model_args)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])         # TODO: 这里需要测试是不是按照预想的方式进行的拼接
                st_local_graph.append(k_order_graph)                # [num_nodes, kernel_size x num_nodes]
        return st_local_graph

    def forward(self, **inputs):
        X   = inputs['X']
        E_d = inputs['E_d']
        E_u = inputs['E_u']
        T_D = inputs['T_D']
        D_W = inputs['D_W']
        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        # mask
        dist_mx = self.mask(dist_mx)
        # normalization
        dist_mx = self.normalizer(dist_mx)
        # multi order
        mul_mx  = self.multi_order(dist_mx)
        # spatial temporal localization (for subsequential computation) 静态图的muti-order和st localization放在一起了，在`Spa_block.get_graph`中。
        dynamic_graphs = self.st_localization(mul_mx)
        
        return dynamic_graphs
