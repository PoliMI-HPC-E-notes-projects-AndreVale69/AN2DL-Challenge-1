"""
Static Multi-Layer Perceptron (MLP) Block.
"""
from torch import Tensor
from torch.nn import Module, Sequential, Linear, RReLU, Dropout


class StaticMLP(Module):
    """
    Static Multi-Layer Perceptron (MLP) Block.

    This block consists of two fully connected layers with RReLU activation and Dropout in between.
    It is designed to process static input features of dimension 7 and output a feature vector of dimension 64.
    """
    def __init__(self, in_dim: int = 7, hidden: int = 64, dropout: float = 0.3):
        """
        Static Multi-Layer Perceptron (MLP) Block.

        This block consists of two fully connected layers with RReLU activation and Dropout in between.
        It is designed to process static input features of dimension `in_dim` and output a feature vector of dimension `hidden`.
        :param in_dim: The dimension of the input features. Default is 7.
        :param hidden: The dimension of the hidden layer and output features. Default is 64.
        :param dropout: The dropout rate applied after each activation. Default is 0.3.
        """
        super().__init__()
        self.net = Sequential(
            # First layer transforms input features to hidden dimension
            Linear(in_dim, hidden), RReLU(0.1, 0.3), Dropout(dropout),
            # Second layer maintains hidden dimension1
            Linear(hidden, hidden), RReLU(0.1, 0.3), Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Static MLP block.

        Simply passes the input through the defined sequential network.
        :param x: Input tensor of shape (batch_size, in_dim).
        :return: Output tensor of shape (batch_size, hidden).
        """
        return self.net(x)