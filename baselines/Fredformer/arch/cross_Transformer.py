import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
import math

class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head *  heads 
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k
        
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out),attn
       
    


class c_Transformer(nn.Module):           ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim,  heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) 
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x_n,attn=attn(x)
            x = x_n + x
            x = ff(x) + x
        return x,attn





class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout , patch_dim, horizon, d_model):
        super().__init__()
        
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim),nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.mlp_head = nn.Linear(dim, d_model)#horizon)
        

    def forward(self, x):
        
        x = self.to_patch_embedding(x)
        x,attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x#,attn