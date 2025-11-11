"""
Temporal Convolutional Network (TCN) implementation using dilated convolutions.
"""
from torch import Tensor
from torch.nn import Module, ModuleList

from internal.nn.blocks.convolution_1d import Convolution1d


class TCN(Module):
    """
    Temporal Convolutional Network with dilated convolutions.

    The TCN consists of a series of Conv1dBlock layers with increasing dilation rates.
    This architecture allows the network to capture long-range dependencies in sequential data.
    """
    def __init__(
            self,
            channels: int,
            dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
            kernel_size: int = 5,
            dropout: float = 0.3
    ):
        """
        Temporal Convolutional Network with dilated convolutions.

        This network is composed of multiple Conv1dBlock layers, each with a specified dilation rate.
        The dilation rates allow the network to have a larger receptive field, enabling it to model
        long-term dependencies in sequential data.
        :param channels: Number of input and output channels for each Conv1dBlock.
        :param dilation_rates: Tuple of dilation rates for each Conv1dBlock. Defaults to (1, 2, 4, 8).
        :param kernel_size: Size of the convolutional kernel in each Conv1dBlock.
        :param dropout: Dropout rate applied in each Conv1dBlock.
        """
        super().__init__()
        # Initialize a list of Conv1dBlock layers with specified dilation rates
        self.blocks: ModuleList[Convolution1d] = ModuleList([
            # Create a Conv1dBlock for each dilation rate
            Convolution1d(channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout) for dilation in dilation_rates
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TCN.

        Applies each Conv1dBlock in sequence to the input tensor.
        :param x: Input tensor of shape (batch_size, channels, length).
        :return: Output tensor of the same shape as input.
        """
        for block in self.blocks:
            # Forward of each Conv1dBlock
            x: Tensor = block(x)
        return x