from torch import nn

ACT2FN = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "prelu": nn.PReLU(),
    "gelu": nn.GELU(),
}
