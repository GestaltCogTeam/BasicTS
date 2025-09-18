import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from argparse import Namespace

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class QueryAdaptiveMasking(nn.Module):
    def __init__(self, dim=1, start_prob =0.1, end_prob =0.5):
        super().__init__()
        self.dim = dim
        self.start_prob = start_prob
        self.end_prob = end_prob
    def forward(self, x):
        if not self.training:
            return x
        else:
            size = x.shape[self.dim]
            dropout_prob = torch.linspace(self.start_prob,self.end_prob,steps=size,device=x.device).view([-1 if i == self.dim else 1 for i in range(x.dim())])
            mask = torch.bernoulli(1 - dropout_prob).expand_as(x)
            return x*mask

class Model_backbone(nn.Module):
    def __init__(self, c_in:int, seq_len:int, pred_len:int, patch_len:int=24, stride:int=24, n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, 
                 attn_dropout:float=0., dropout:float=0., res_attention:bool=True, independence:bool=False, store_attn:bool=False, QAM_start:float = 0.1, 
                 QAM_end:float =0.5, padding_patch = None):
        
        super().__init__()
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        pred_patch_num = (pred_len+patch_len-1)//patch_len
        seq_patch_num = int((seq_len - patch_len)/stride + 1)
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            seq_patch_num += 1
        
        # Backbone 
        self.backbone = Dummy_Embedding(c_in, seq_patch_num=seq_patch_num, patch_len=patch_len, pred_patch_num=pred_patch_num,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, QAM_start=QAM_start, QAM_end=QAM_end,
                                res_attention=res_attention, independence = independence, store_attn=store_attn)

        self.n_vars = c_in
        self.pred_len = pred_len
        self.proj = Projection(d_model, patch_len)
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        mean = z.mean(2, keepdim=True)
        std = torch.sqrt(torch.var(z, dim=2, keepdim=True, unbiased=False) + 1e-5)
        z = (z - mean)/std 
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x seq_patch_num x patch_len]

        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x pred_patch_num x d_model]
        z = self.proj(z)                                                                    # z: [bs x nvars x pred_len] 
        
        # denorm
        z = z * (std[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        z = z + (mean[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        
        return z


class Projection(nn.Module):
    def __init__(self, d_model, patch_len):
        super().__init__()
        self.linear = nn.Linear(d_model,patch_len)
        self.flatten = nn.Flatten(start_dim = -2)
            
    def forward(self, x):                         
        x = self.linear(x)
        x = self.flatten(x)
        return x
    
class Dummy_Embedding(nn.Module): 
    def __init__(self, c_in, seq_patch_num, patch_len, pred_patch_num, n_layers=3, d_model=128, n_heads=16, QAM_start = 0.1, QAM_end =0.5,
                 d_ff=256, attn_dropout=0., dropout=0., store_attn=False, res_attention=True, independence = False):
             
        super().__init__()
        
        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)      
        self.dropout = nn.Dropout(dropout)
        # Dummy Input
        self.independence = independence
        if self.independence:
            self.dummies = nn.Parameter(0.5*torch.randn(pred_patch_num, patch_len))
        else:
            self.dummies = nn.Parameter(0.5*torch.randn(c_in, pred_patch_num, patch_len))
        self.independence = independence
        # Positional encoding
        self.PE = nn.Parameter(0.04*torch.rand(seq_patch_num, d_model)-0.02)
        # Encoder
        self.decoder = Decoder(seq_patch_num, d_model, n_heads, pred_patch_num, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                               QAM_start=QAM_start, QAM_end=QAM_end, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        
    def forward(self, x) -> Tensor:                                           # x: [bs x nvars x seq_patch_num x patch_len]
        bs = x.shape[0]
        n_vars = x.shape[1]
        # Input encoding
        x = self.W_P(x) + self.PE                                             # x: [bs x nvars x seq_patch_num x d_model]
        dummies = self.W_P(self.dummies)                                      # dummies: [bs x nvars x pred_patch_num x d_model]
        x = torch.reshape(x, (bs*n_vars,x.shape[2],x.shape[3]))               # x: [bs * nvars x seq_patch_num x d_model]
        
        seq_patch = self.dropout(x)                                           # seq_patch: [bs * nvars x patch_num x d_model]
         
        if self.independence:
            pred_patch = dummies.unsqueeze(0).repeat(bs*n_vars,1,1)           
        else:
            pred_patch = dummies.unsqueeze(0).repeat(bs,1,1,1)                
            pred_patch = torch.reshape(pred_patch, (bs*n_vars,pred_patch.shape[2],pred_patch.shape[3]))  # pred_patch: [bs * nvars x pred_patch_num x d_model]
        
        # decoder
        z = self.decoder(seq_patch, pred_patch)                               # z: [bs * nvars x pred_patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))             # z: [bs x nvars x pred_patch_num x d_model]
        return z       
            
# Cell
class Decoder(nn.Module):
    def __init__(self, seq_patch_num, d_model, n_heads, pred_patch_num, d_ff=None, attn_dropout=0., dropout=0., QAM_start = 0.1, QAM_end =0.5,
                        res_attention=False, n_layers=1, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(seq_patch_num, d_model, pred_patch_num, n_heads=n_heads, d_ff=d_ff, QAM_start=QAM_start, QAM_end=QAM_end,
                                                      attn_dropout=attn_dropout, dropout=dropout, res_attention=res_attention,
                                                      store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, seq:Tensor, pred:Tensor):
        scores = None
        if self.res_attention:
            for mod in self.layers: seq, pred, scores = mod(seq, pred, prev=scores)
            return pred
        else:
            for mod in self.layers: seq, pred = mod(seq, pred)
            return pred

class DecoderLayer(nn.Module):
    def __init__(self, seq_patch_num, d_model, pred_patch_num, n_heads, d_ff=256, store_attn=False, QAM_start = 0.1, QAM_end =0.5,
                 attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        # Multi-Head attention
        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        self.norm_attn = nn.LayerNorm(d_model)
        # Position-wise Feed-Forward
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                GEGLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff//2, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.store_attn = store_attn

    def forward(self, seq:Tensor, pred:Tensor, prev=None) -> Tensor:
        # pred_patch: [bs * nvars x pred_patch_num x d_model]
        ## Multi-Head attention
        if self.res_attention:
            pred2, attn, scores = self.cross_attn(pred, seq, seq, prev)
        else:
            pred2, attn = self.cross_attn(pred, seq, seq)
        if self.store_attn:
            self.attn = attn
        pred = pred + self.dropout_attn(pred2)    
        pred = self.norm_attn(pred)
        
        pred2 = self.ffn(pred)
        pred = pred + self.dropout_ffn(pred2)     
        pred = self.norm_ffn(pred)
        
        if self.res_attention: return seq, pred, scores
        else: return seq, pred

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x pred_patch_num x d_model]
            K, V:    [batch_size (bs) x seq_patch_num x d_model]
        """
        super().__init__()
        d_h = d_model // n_heads

        self.scale = d_h**-0.5
        self.n_heads, self.d_h = n_heads, d_h

        self.W_Q = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_h, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):

        bs = Q.size(0)
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_h)     
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_h) 
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_h) 

        attn_scores = torch.einsum('bphd, bshd -> bphs', q_s, k_s) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bphs, bshd -> bphd', attn_weights, v_s)
        output = output.contiguous().view(bs, -1, self.n_heads*self.d_h)
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class CATS(nn.Module):
    def __init__(self, **model_args):
        """
            Paper: Are Self-Attentions Effective for Time Series Forecasting?
            Link: https://arxiv.org/pdf/2405.16877
            Official Code: https://github.com/dongbeank/CATS
            Venue:  NIPS 2024
            Task: Long-term Time Series Forecasting
        """
        
        super().__init__()
        args = Namespace(**model_args)
        # load parameters
        c_in = args.dec_in
        seq_len = args.seq_len
        self.pred_len = args.pred_len
        n_layers = args.d_layers
        n_heads = args.n_heads
        d_model = args.d_model
        d_ff = args.d_ff
        dropout = args.dropout
        independence = args.query_independence
        patch_len = args.patch_len
        stride = args.stride
        padding_patch = args.padding_patch
        store_attn = args.store_attn

        QAM_start = args.QAM_start
        QAM_end = args.QAM_end

        self.model = Model_backbone(c_in=c_in, seq_len = seq_len, pred_len=self.pred_len, patch_len=patch_len, stride=stride, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,independence=independence, 
                                    store_attn=store_attn, padding_patch = padding_patch, QAM_start=QAM_start, QAM_end=QAM_end)

    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of CATS.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        
        x = history_data[..., 0].permute(0,2,1)        # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)        # x: [Batch, Input length, Channel]

        return x[:,:self.pred_len,:].unsqueeze(-1)
