"""
1D Convolutional Block with Residual Connection.
"""
from torch import Tensor
from torch.nn import Module, Conv1d, GroupNorm, Dropout
from torch.nn.functional import gelu


class Convolution1d(Module):
    """
    1D Convolutional Block with Residual Connection.
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.3):
        """
        1D Convolutional Block with Residual Connection.

        This block consists of a 1D convolutional layer followed by Group Normalization,
        GELU activation, and Dropout. A residual connection adds the input to the output
        of these operations.
        :param channels: Number of input and output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param dilation: Dilation rate for the convolution.
        :param dropout: Dropout rate.
        """
        super().__init__()
        # Define layers where the padding is set to maintain the input length;
        # the formula is: padding = (kernel_size - 1) * dilation // 2
        self.conv = Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) * dilation // 2,
                           dilation=dilation)
        self.norm = GroupNorm(1, channels)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Convolution 1D.

        Applies convolution, normalization, activation, and dropout, then adds the input
        as a residual connection.
        :param x: Input tensor of shape (batch_size, channels, length).
        :return: Output tensor of the same shape as input.
        """
        # The residual is simply the input x because input and output shapes are the same
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = gelu(x)
        x = self.drop(x)
        return x + residual
