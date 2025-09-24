import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        #todo time embedding
        self.timeconv = nn.ModuleList()

        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
        #todo time embedding
        for kern in self.kernel_set:
            self.timeconv.append(nn.Conv1d(4,4,kern,dilation=(dilation_factor)))
        self.timepro = nn.Conv1d(16, 4, 1)

    def forward(self,input,x_mark_enc):
        x = []
        #todo time embedding
        x_mark_enc = x_mark_enc.transpose(-1, -2)
        x_mark_enc_list = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
            x_mark_enc_list.append(self.timeconv[i](x_mark_enc))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
            x_mark_enc_list[i] = x_mark_enc_list[i][..., -x_mark_enc_list[-1].size(2):]
        x = torch.cat(x,dim=1)
        x_mark_enc = torch.cat(x_mark_enc_list, dim=1)
        x_mark_enc = self.timepro(x_mark_enc)
        return x,x_mark_enc.transpose(-2,-1)
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout: float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x,x_mark_enc):
        _filter,x_mark_enc_red = self.filter_conv(x,x_mark_enc)
        filter = torch.tanh(_filter)
        _gate,x_mark_enc_red = self.gate_conv(x,x_mark_enc)
        gate = torch.sigmoid(_gate)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=self.training)
        return x,x_mark_enc_red

