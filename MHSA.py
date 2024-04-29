import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from RotaryPE import apply_rotary_pe
from einops import rearrange


class RelativeMultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_head=16, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "Dimension of model not divisible by num heads"
        self.d_model = d_model
        self.n_heads = n_head
        self.head_dim = int(d_model / n_head)
        self.sqrt_dim = math.sqrt(self.head_dim)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, freqs_complex: torch.tensor, mask=None):
        # get qkv (batch, time, dim) -> (batch, time, dim)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # print("q.shape: ", q.shape)

        q = apply_rotary_pe(q, freqs_complex, q.device)
        k = apply_rotary_pe(k, freqs_complex, k.device)

        # split qkv -> (batch, time, dim) -> (batch, time, n_heads, head_dim)
        q = rearrange(q, "b t (n_heads head_dim) -> b n_heads t head_dim", n_heads=self.n_heads)
        k = rearrange(k, "b t (n_heads head_dim) -> b n_heads head_dim t", n_heads=self.n_heads)
        v = rearrange(v, "b t (n_heads head_dim) -> b n_heads t head_dim", n_heads=self.n_heads)
        # print("q: ", q.shape)
        # print("k: ", k.shape)
        # print("v: ", v.shape)

        attention = torch.matmul(q, k) / self.sqrt_dim
        # print("attention: ", attention.shape)

        attention = F.softmax(attention, -1)
        attention = self.dropout(attention)

        if mask is not None:
            attention.masked_fill(mask, -1e9)

        context = torch.matmul(attention, v)
        # print("applied attention to v: ", context.shape)

        context = rearrange(context.contiguous(), "b n_h t h_dim -> b t (n_h h_dim)", n_h=self.n_heads)
        # print("rearrange context: ", context.shape)

        return self.wo(context)




