import math
from typing import Optional

from torch import nn


class BaseAttention(nn.Module):
    """
    Base module to implement various self-attention mechanisms
    Allows for (Q,K) and V to have different dimensions
    """

    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
        # eps: float = 1e-6,
    ):
        super().__init__()

        # By default all matrices have the same shape
        if qk_dim is None:
            qk_dim = embed_dim

        if embed_dim % nb_heads != 0 or qk_dim % nb_heads != 0:
            raise ValueError(
                "Embed dim and QK dim (if specified) mus tbe divisible by the number of heads.\n"
                f"Embed: {embed_dim}, QK: {qk_dim} -> n_heads: {nb_heads}"
            )

        # Dimensions and parameters
        self.embed_dim = embed_dim
        self.qk_dim = qk_dim
        self.nb_heads = nb_heads
        self.dropout = dropout

        self.head_dim = embed_dim // nb_heads
        self.head_qk_dim = qk_dim // nb_heads

        # Projectors
        self.k_proj = nn.Linear(embed_dim, qk_dim)
        self.q_proj = nn.Linear(embed_dim, qk_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)


class MultiHeadAttention(BaseAttention):
    """
    Our implementation of standard scaled dot product self-attention
    as in "Attention is all you need"
    """

    def __init__(self, nb_heads: int, embed_dim: int, dropout: float = 0):
        super().__init__(nb_heads, embed_dim, None, dropout)

    def forward(self, input):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        sqrt_dim = math.sqrt(self.head_dim)
        atn_logits = q @ k.transpose(-1, -2) / sqrt_dim
        atn_probs = nn.functional.softmax(atn_logits, dim=-1)

        V = atn_probs @ v

        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.out_proj(self.proj_drop(V))

        return out


class LinearKernelAttention(BaseAttention):
    """
    Implementation of the Linear Kernel Attention from:
    doi.org/10.48550/arXiv.2006.16236
    """

    def __init__(
        self, nb_heads: int, embed_dim: int, dropout: float = 0, eps: float = 1e-6
    ):
        super().__init__(nb_heads, embed_dim, None, dropout)

        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, input):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        KtV = k.transpose(-1, -2) @ v

        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        Z = Z.expand(batch_size, nb_row, self.nb_heads, nb_col, self.head_dim)

        V = Z * (q @ KtV)
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out


class ScaledLinearAttention(BaseAttention):
    """
    Custom version of the Linear Kernel Attention with dimension 1 for Q and K.
    """

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(nb_heads, embed_dim, nb_heads, dropout)

        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, input):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Scale Q to keep amplitude under control
        q = q / q.mean(dim=-2, keepdim=True)

        # Normalize K
        k = k / k.sum(
            dim=-2, keepdim=True
        )  # Sum directly on -2 instead of transposing an summing

        KtV = k.transpose(-1, -2) @ v

        V = q @ KtV
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out
