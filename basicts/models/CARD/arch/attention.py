import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False):
        super().__init__()

        self.over_hidden = over_hidden
        self.untoken = untoken
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        
        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        self.norm_post2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                                )     
        self.merge_size = config.merge_size

        ema_size = max(config.enc_in,config.total_token_number,config.dp_rank)
        ema_matrix = torch.zeros((ema_size,ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1,config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix',ema_matrix)


    def ema(self,src):
        return torch.einsum('bnhad,ga ->bnhgd',src,self.ema_matrix[:src.shape[-2],:src.shape[-2]])

    def ema_trianable(self,src):
        alpha = F.sigmoid(self.alpha)
        weights = alpha * (1 - alpha) ** self.arange[-src.shape[-2]:]
        w_f = torch.fft.rfft(weights,n = src.shape[-2]*2)
        src_f = torch.fft.rfft(src.float(),dim = -2,n = src.shape[-2]*2)    
        src_f = (src_f.permute(0,1,2,4,3)*w_f)
        src1 =torch.fft.irfft(src_f.float(),dim = -1,n=src.shape[-2]*2)[...,:src.shape[-2]].permute(0,1,2,4,3)#.half()
        return src1

    def dynamic_projection(self,src,mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp,dim = -1)
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf',src,src_dp)
        return src_dp

    def forward(self, src):
        B,nvars, H, C, = src.shape
        
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
   
        q, k, v = qkv[0], qkv[1], qkv[2]
    
        if not self.over_hidden: 
        
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k))/ self.head_dim ** -0.5
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)

        else:
            v_dp,k_dp = self.dynamic_projection(v,self.dp_v) , self.dynamic_projection(k,self.dp_k)
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp))/ self.head_dim ** -0.5
            

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k)/ q.shape[-2] ** -0.5
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)

        merge_size = self.merge_size
        if not self.untoken:
            output1 = rearrange(output_along_token.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        
        
            output2 = rearrange(output_along_hidden.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        

        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)

        src2 =  self.ff_1(output1)+self.ff_2(output2)

        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        return src