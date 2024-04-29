import torch.nn as nn
from Modules import ResidualConnectionModule, FeedForwardModule, ResidualConnectionModuleMHA, MultiHeadAttentionModule
from Convolution import Conv2dSubsampling, ConvolutionModule
from RotaryPE import precompute_theta_freqs


class ConformerBlock(nn.Module):

    def __init__(self,
                 model_dim: int = 512,
                 linear_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 half_step_residual: bool = True,
                 ff_dropout_p: float = 0.1,
                 mha_dropout_p: float = 0.1,
                 conv_dropout_p: float = 0.1,
                 n_heads: int = 16,
                 kernel_size: int = 31
                 ):
        super().__init__()

        if half_step_residual:
            self.linear_residual_factor = 0.5
        else:
            self.linear_residual_factor = 1

        self.ffm1 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=model_dim,
                expansion_factor=linear_expansion_factor,
                dropout=ff_dropout_p,
            ),
            module_factor=self.linear_residual_factor
        )
        self.mham = ResidualConnectionModuleMHA(
            module=MultiHeadAttentionModule(model_dim, n_heads, mha_dropout_p),
            module_factor=1
        )
        self.convm = ResidualConnectionModule(
            module=ConvolutionModule(model_dim, conv_expansion_factor, kernel_size, conv_dropout_p),
            module_factor=1
        )

        self.ffm2 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=model_dim,
                expansion_factor=linear_expansion_factor,
                dropout=ff_dropout_p,
            ),
            module_factor=self.linear_residual_factor
        )

    def forward(self, x, freqs_complex, mask=None):
        x = self.ffm1(x)
        x = self.mham(x, freqs_complex, mask)
        x = self.convm(x)
        x = self.ffm2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, input_dim: int = 64, model_dim: int = 512, input_dropout: float = 0.1):
        super().__init__()
        self.conv2dsubsample = Conv2dSubsampling(1, model_dim)
        self.input_projection = nn.Sequential(
            nn.Linear((((input_dim - 1) // 2 - 1) // 2 * model_dim), model_dim),
            nn.Dropout(input_dropout)
        )

    def forward(self, x, input_lengths):
        # print("Original:", x.shape)
        x, output_lengths = self.conv2dsubsample(x, input_lengths)  # (batch, time, dim)
        # print("Conv2d: ", x.shape)
        x = self.input_projection(x)  # (batch, time, model_dim)
        # print("Input_Projection:", x.shape)
        return x, output_lengths


class Conformer(nn.Module):

    def __init__(self, input_dim: int, model_dim: int,
                 num_classes: int,
                 linear_expansion_factor: int = 4,
                 conv_expansion_factor: int = 2,
                 half_step_residual: bool = True,
                 n_heads: int = 16,
                 kernel_size: int = 31,
                 dropout: float = 0.1,
                 num_conformer_blocks: int = 4
                 ):
        super().__init__()

        self.model_dim = model_dim
        self.encoder = Encoder(input_dim, model_dim, dropout)
        self.layers = nn.ModuleList([ConformerBlock(
            model_dim,
            linear_expansion_factor,
            conv_expansion_factor,
            half_step_residual,
            dropout,
            dropout,
            dropout,
            n_heads,
            kernel_size
        ) for _ in range(num_conformer_blocks)])
        self.output_projection = nn.Linear(model_dim, num_classes)


    def forward(self, x, input_lengths):
        x, output_lengths = self.encoder(x, input_lengths)

        freqs_complex = precompute_theta_freqs(self.model_dim, x.size(1), "cpu")

        for layer in self.layers:
            x = layer(x, freqs_complex)
        x = self.output_projection(x)
        x = nn.functional.log_softmax(x, dim=-1)
        # print("output shape: ", x.shape)
        return x, output_lengths
