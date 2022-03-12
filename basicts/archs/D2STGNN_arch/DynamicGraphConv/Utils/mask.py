import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, method, **model_args):
        super().__init__()
        self.method = method
        if self.method == 'geograph':
            adj_list  = model_args['adjs']
            self.mask  = adj_list
        elif self.method == 'geograph01':
            self.order  = model_args['order']
            # TODO: 暂时先不用扩散，先把一阶的时候的情况给弄明白就可以了
            adj_list  = model_args['adjs']
            self.mask = [torch.where(tmp != 0, torch.ones_like(tmp), torch.zeros_like(tmp)) for tmp in adj_list]
        elif self.method == 'identity':
            self.mask = torch.ones_like(model_args['adjs_ori'])
        elif self.method == 'eye':
            self.mask   = (1 - torch.eye(model_args['num_nodes'])).to(model_args['device'])
        elif self.method == 'topk':
            # topk mask is dynamic
            self.k = 20
            print("使用TopK Mask，请确认激活函数正确。K大小为:{0}".format(self.k))
        else:
            raise Exception("Unknown mask!")
    
    def _mask(self, index, adj):
        if self.method in "geograph01":
            mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7      # TODO: 测试删掉无穷小
            return mask.to(adj.device) * adj
        if self.method != 'topk':
            self.mask = self.mask + torch.ones_like(self.mask) * 1e-7
        else:
            adj_abs = torch.abs(adj)
            mask = torch.zeros(adj_abs.size(0), adj_abs.size(1), adj_abs.size(2)).to(adj[0].device)
            mask.fill_(float('0'))
            s1,t1 = (adj_abs + torch.rand_like(adj_abs)*0.01).topk(self.k,2)
            mask.scatter_(1,t1,s1.fill_(1))
            self.mask = mask + torch.ones_like(adj_abs) * 1e-7
        return self.mask * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
