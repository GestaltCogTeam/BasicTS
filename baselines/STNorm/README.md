The STNorm in BasicTS follows its original implementation; however, this hyperparameter design may lead to some misunderstandings.

```python
class STNorm(nn.Module):
    """
    Paper: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
    Link: https://dl.acm.org/doi/10.1145/3447548.3467330
    Ref Official Code: https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py
    Venue: SIGKDD 2021
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, num_nodes, tnorm_bool, snorm_bool, in_dim, out_dim, channels, kernel_size, blocks, layers):
```

 Here’s a further clarification: 
 - `in_dim` and `out_dim` are not the same concept.
 - `in_dim` refers to the number of input channels, specifically the `C` in `[B, L, N, C]`. 
 - Meanwhile, `out_dim` actually represents the output length, which corresponds to the `L` in `[B, L, N, C]`.
 - Additionally, STNorm does not implicitly specify the input length. 
- Instead, it controls the receptive field (i.e., the effective input length) by adjusting the number of convolutional layers and the size of the convolutional kernels.
