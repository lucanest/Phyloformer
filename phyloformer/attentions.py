"""The attentions module contains classes for the MultiHeadAttention modules 
in the Phyloformer network
"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class KernelAxialMultiAttention(nn.Module):
    def __init__(self, h_dim, n_heads, dropout=0.0, eps=1e-6, n=None, k=None):
        super().__init__()

        self.n_heads = n_heads
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.elu = nn.ELU()
        self.eps = eps

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Mask is in pair representation space so (batch_size *  n_pairs)
        # Input is of shape (batch_size * n_pairs * seq_len * h_dim)

        print("\t\tIn Attention:")
        print(f"\t\t Input: {x.shape}")
        Bs = x.shape[0]

        # M:n_pairs, T:seq_len, C:h_dim, N:n_heads, D:h_dim/n_heads (For row attn)
        # For col attn switch M and T
        M, T, C = x.shape[-3:]
        N, D = self.n_heads, C // self.n_heads

        q = self.q_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        k = self.k_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        v = self.v_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        print(f"\t\tq: {q.shape}, k: {k.shape}, v: {v.shape}")

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Transformers are RNNs linear attention paper adds the mask to the
        # K matrix in their implementation:
        # https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/attention/linear_attention.py#L67
        # k   : Row(Bs * M * N * T * D), Col(Bs * T * N * M * D)
        # mask: (Bs * M)
        if mask is not None:
            print(f"\t\t mask: {mask.shape}")
            if mask.shape[-1] == M:  # Row attention
                mask = mask[:, :, None, None, None]
            else:  # Col attention
                mask = mask[:, None, None, :, None]
            print(f"\t\t mask..: {mask.shape}")
            print(f"\t\t k:      {k.shape}")

            k = k * mask

        KtV = k.transpose(-1, -2) @ v
        print(f"\t\tKtV: {KtV.shape}")
        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        print(f"\t\tZ: {Z.shape}")
        Z = Z.expand(Bs, M, N, T, D)
        print(f"\t\tZ..: {Z.shape}")
        V = Z @ KtV
        print(f"\t\tV: {V.shape}")
        V = V.transpose(2, 3).contiguous().view(Bs, -1, T, N * D)
        print(f"\t\tV..: {V.shape}")

        a = None
        out = self.proj_drop(self.proj_net(V))
        return out, a
