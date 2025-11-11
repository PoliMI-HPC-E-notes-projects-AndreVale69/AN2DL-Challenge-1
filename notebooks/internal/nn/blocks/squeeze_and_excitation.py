"""
Squeeze-and-Excitation (SE) block for 1D signals.
Implements channel-wise attention mechanism to recalibrate feature maps.
"""
from torch import Tensor
from torch.nn import Module, Sequential, ReLU, Sigmoid, Linear


class SE1D(Module):
    """
    Squeeze-and-Excitation (SE) block for 1D signals.
    """
    def __init__(self, channels: int, reduction_ratio: int = 8):
        """
        Squeeze-and-Excitation block for 1D signals.

        This block implements channel-wise attention by first squeezing the input tensor
        along the length dimension to create a channel descriptor, then passing this descriptor
        through a two-layer fully connected network to learn channel-wise weights, and finally
        scaling the original input tensor by these weights.
        :param channels: Number of channels
        :param reduction_ratio: Reduction ratio
        """
        super().__init__()
        self.fc: Sequential = Sequential(
            # First layer reduces the channel dimension by factor of r
            Linear(channels, max(1, channels // reduction_ratio)), ReLU(inplace=True),
            # Second layer restores the channel dimension to c
            Linear(max(1, channels // reduction_ratio), channels), Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SE block. It applies channel-wise attention to the input tensor.

        The block first computes the mean across the length dimension to obtain a channel descriptor,
        then passes this descriptor through a two-layer fully connected network with a ReLU activation
        in between, and finally applies a sigmoid activation to obtain channel-wise weights.
        These weights are then used to scale the original input tensor channel-wise.
        :param x: Input tensor of shape (batch_size, channels, length)
        :return: Output tensor of the same shape as input, with channel-wise attention applied.
        """
        return x * self.fc(x.mean(dim=2)).unsqueeze(-1)
