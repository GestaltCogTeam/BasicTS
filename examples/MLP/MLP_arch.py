import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(history_seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, prediction_seq_len)
        self.act = nn.ReLU()

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """

        history_data = history_data[..., 0].transpose(1, 2)     # B, N, L
        prediction = self.fc2(self.act(self.fc1(history_data))).transpose(1, 2)     # B, L, N
        return prediction.unsqueeze(-1)         # B, L, N, C
