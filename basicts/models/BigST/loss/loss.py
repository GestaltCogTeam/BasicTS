import torch
import numpy as np
from basicts.metrics import masked_mae

def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    
    node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]] # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start) # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2) # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)
    
    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2) # [B, E, 1]
    attn_score = attn1 / attn2 # [B, E, 1]
    
    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    spatial_loss = torch.mean(attn_score.log() * d_norm)
    
    return spatial_loss

def bigst_loss(prediction, target, node_vec1, node_vec2, supports, use_spatial):
    if use_spatial:
        supports = [support.to(prediction.device) for support in supports]
        edge_indices = torch.nonzero(supports[0] > 0)
        s_loss = spatial_loss(node_vec1, node_vec2, supports, edge_indices)
        return masked_mae(prediction, target, 0.0) - 0.3 * s_loss # 源代码：pipline.py line30
    else:
        return masked_mae(prediction, target, 0.0)