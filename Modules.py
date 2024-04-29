from MHSA import RelativeMultiHeadAttention
import torch.nn as nn


class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x):
        return self.module(x) * self.module_factor + x * self.input_factor


class ResidualConnectionModuleMHA(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super().__init__()
        self.mha = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x, freqs_complex, mask=None):
        return self.mha(x, freqs_complex, mask) * self.module_factor + x * self.input_factor


class FeedForwardModule(nn.Module):

    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.sequence(x)
        # print("feedforward: ", x.shape)
        return x


class MultiHeadAttentionModule(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_complex, mask=None):
        x = self.layer_norm(x)
        x = self.mha(x, x, x, freqs_complex, mask)
        x = self.dropout(x)
        return x


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)