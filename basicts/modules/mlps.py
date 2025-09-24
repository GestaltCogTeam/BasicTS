import torch
from torch import nn

from .activations import ACT2FN


class MLPLayer(nn.Module):

    """
    MLP layer.
    """
    def __init__(self,
        input_size: int,
        intermediate_size: int,
        output_size: int = None,
        hidden_act: str = "relu",
        dropout: float = 0.0,
        bias: bool = True) -> None:
        """
        Initialize the MLP layer.

        Args:
            input_size (int): Input feature dimension.
            intermediate_size (int): Hidden dimension.
            output_size (int): Output feature dimension.
            hidden_act (str): Activation function.
        """
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.act_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [..., input_size]

        Returns:
            torch.Tensor: Output tensor of shape [..., output_size]
        """
        return self.fc2(self.dropout(self.act_fn(self.fc1(inputs))))


class ResMLPLayer(nn.Module):

    """
    MLP layer with residual connection.
    """
    def __init__(self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "relu",
        dropout: float = 0.0) -> None:
        """
        Initialize the MLP layer.

        Args:
            hidden_size (int): Input hidden dimension.
            intermediate_size (int): Intermediate hidden dimension.
            hidden_act (str): Activation function.
        """
        super().__init__()
        self.act_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [..., input_size]

        Returns:
            torch.Tensor: Output tensor of shape [..., output_size]
        """
        return self.fc2(self.dropout(self.act_fn(self.fc1(inputs)))) + inputs
