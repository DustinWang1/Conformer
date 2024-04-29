import torch.nn as nn
from Modules import Transpose


class Conv2dSubsampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, dim, time): Tensor containing sequence of inputs

    Returns: outputs, output_lengths (REDACTED UNTIL USEFUL)
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.convSeq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def forward(self, x, input_lengths):
        # (batch, in_channel, freq, time)
        x = self.convSeq(x)  # (batch, out_channel, subsampled_freq, subsampled_time)
        batch_size, out_channel, dim, length = x.shape
        x = x.contiguous().view(batch_size, out_channel * dim, length)  # (batch, dim, time)
        x = x.permute(0, 2, 1)  # (batch, time, dim)

        # Divides the sequence length by 4
        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return x, output_lengths


class PointwiseConv1d(nn.Module):
    # 1x1 kernel convolution practically

    def __init__(self, in_channels, out_channels, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv1d(nn.Module):
    # groups = in_channels
    def __init__(self, in_channels, out_channels, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvolutionModule(nn.Module):

    def __init__(self, in_channels: int, expansion_factor: int = 2, kernel_size: int = 31, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        # print("After convolution: ", x.shape)
        return x
