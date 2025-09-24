import torch


def _get_top_k_indices(stats_tensor: torch.Tensor, K: float) -> torch.Tensor:
        
    # 拆分均值和方差
    mean = stats_tensor[:, :, 0]  # [N, C]
    var = stats_tensor[:, :, 1]   # [N, C]

    # 计算分位数阈值
    k_fraction = K / 100.0
    mean_thresholds = torch.quantile(
        mean, 1 - k_fraction, 
        dim=0, keepdim=True,
        interpolation='higher'
    )
    var_thresholds = torch.quantile(
        var, 1 - k_fraction,
        dim=0, keepdim=True,
        interpolation='higher'
    )

    # 生成布尔掩码
    mean_mask = mean >= mean_thresholds
    var_mask = var >= var_thresholds
    union = mean_mask | var_mask
    # indices = torch.nonzero(union, as_tuple=False)
    return union

if __name__ == "__main__":
    stats_tensor = torch.rand((10, 7, 2))
    print(stats_tensor)
    K = 20
    indices = _get_top_k_indices(stats_tensor, K)
    print(indices)
    print(indices.shape)