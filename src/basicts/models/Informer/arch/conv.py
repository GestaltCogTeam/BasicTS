import torch
from torch import nn

from basicts.modules.activations import ACT2FN


class ConvLayer(nn.Module):

    """
    Convolutional layer for the Informer model.
    """

    def __init__(self, num_features: int, hidden_act: str = "elu"):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.downConv = nn.Conv1d(num_features,
                                  num_features,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode="circular")
        self.norm = nn.BatchNorm1d(num_features)
        self.hidden_act = ACT2FN[hidden_act]
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downConv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)
        return x.transpose(1, 2)
