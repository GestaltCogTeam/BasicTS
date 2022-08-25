import torch.nn as nn

class ResidualDecomp(nn.Module):
    r"""
    Residual decomposition.
    """
    def __init__(self, input_shape):
        super().__init__()
        self.ln     = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u
