import torch
from einops import rearrange

# ##
# Computed with the formula given in RoPE
# ##
def precompute_theta_freqs(model_dim, seq_len, device, theta: float = 10000.0):

    assert model_dim % 2 == 0, "Head dimension not divisible by 2 for rotary PE"

    # Calculate theta values by formula
    theta_numerator = torch.arange(0, model_dim / 2).float()
    theta = 1.0 / (theta ** (2 * theta_numerator / model_dim)).to(device)

    m = torch.arange(seq_len, device=device)

    #multiply each theta by each m
    freqs = torch.outer(m, theta).float()

    # turn in to complex number e^(im(theta)) which is cos(x)+isin(x)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    # print("freqs complex: ", freqs_complex.shape)

    return freqs_complex

def apply_rotary_pe(x: torch.tensor, freqs_complex: torch.polar, device: str):
    # x -> (batch, time, dim)

    # turn into -> (batch, time, dim/2)
    x = torch.view_as_complex(rearrange(x, "b t (d two) -> b t d two", two=2))
    # print("complex form: ", x.shape)

    x_rotated = x * freqs_complex

    x_reverted = torch.view_as_real(x_rotated)
    # print("Rotated and reverted back to real form: ", x_reverted.shape)

    x = rearrange(x_reverted, "b t dim2 two -> b t (dim2 two)")
    # print("rearrange back to original shape: ", x.shape)

    return x








