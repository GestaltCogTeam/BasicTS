import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model with two fully connected layers.

    This model is designed to take historical time series data as input and produce future predictions.
    It consists of two linear layers with a ReLU activation in between.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer, which maps the input history sequence to a hidden dimension.
        fc2 (nn.Linear): The second fully connected layer, which maps the hidden dimension to the prediction sequence.
        act (nn.ReLU): The ReLU activation function applied between the two layers.
    """

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int):
        """
        Initialize the MultiLayerPerceptron model.

        Args:
            history_seq_len (int): The length of the input history sequence.
            prediction_seq_len (int): The length of the output prediction sequence.
            hidden_dim (int): The number of units in the hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(history_seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, prediction_seq_len)
        self.act = nn.ReLU()

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            history_data (torch.Tensor): A tensor containing historical data, typically of shape `[B, L, N, C]`.
            future_data (torch.Tensor): A tensor containing future data, typically of shape `[B, L, N, C]`.
            batch_seen (int): The number of batches seen so far during training.
            epoch (int): The current epoch number.
            train (bool): Flag indicating whether the model is in training mode.

        Returns:
            torch.Tensor: The output prediction tensor, typically of shape `[B, L, N, C]`.
        """

        history_data = history_data[..., 0].transpose(1, 2) # [B, L, N, C] -> [B, N, L]

        # [B, N, L] --h=act(fc1(x))--> [B, N, D] --fc2(h)--> [B, N, L] -> [B, L, N]
        prediction = self.fc2(self.act(self.fc1(history_data))).transpose(1, 2)

        # [B, L, N] -> [B, L, N, 1]
        return prediction.unsqueeze(-1)
