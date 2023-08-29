"""The attentions module contains classes for the MultiHeadAttention modules 
in the Phyloformer network
"""

import torch
import torch.nn as nn


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

    def einsum_forward(self, x, mask=None):
        # Get dimensions
        B = x.shape[0]
        M, T, C = x.shape[-3:]
        N, D = self.n_heads, C // self.n_heads

        queries = self.q_net(x).view(B, M, T, N, D).transpose(2, 3)
        keys = self.k_net(x).view(B, M, T, N, D).transpose(2, 3)
        values = self.v_net(x).view(B, M, T, N, D).transpose(2, 3)

        # Apply feature map to queries and keys
        Q = self.elu(queries) + 1
        K = self.elu(keys) + 1

        # Apply padding mask if it exists
        if mask is not None:
            # mask shape is [B,M,T], K shape is [B,M,N,T,D]
            K = K * mask[:, :, None, :, None]

        # Compute the KV matrix (in einsum d == e)
        KV = torch.einsum("bmntd,bmnte->bmnde", K, values)

        # Compute normalizer
        Z = 1 / (torch.einsum("bmntd,bmnd->bmnt", Q, K.sum(-2)) + self.eps)

        # Compute the new values
        V = torch.einsum(
            "bmntd,bmnde->bmtne", Z.unsqueeze(-1).expand(B, M, N, T, D), KV
        )
        # reshape values to concatenate along attention heads
        V = V.contiguous().view(B, -1, T, N * D)

        return V.contiguous()

    def forward(self, x, mask=None):
        Bs = x.shape[0]

        # M:n_pairs, T:seq_len, C:h_dim, N:n_heads, D:h_dim/n_heads (For row attn)
        # For col attn switch M and T
        M, T, C = x.shape[-3:]
        N, D = self.n_heads, C // self.n_heads

        q = self.q_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        k = self.k_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        v = self.v_net(x).view(Bs, M, T, N, D).transpose(2, 3)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Transformers are RNNs linear attention paper adds the mask to the
        # K matrix in their implementation:i
        # https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/attention/linear_attention.py#L67
        # mask: (Bs * M * T)
        if mask is not None:
            mask = mask[:, :, None, :, None]

            k = k * mask

        KtV = k.transpose(-1, -2) @ v
        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        Z = Z.expand(Bs, M, N, T, D)
        V = Z @ KtV  # Potentially missing a Q term here ?
        V = V.transpose(2, 3).contiguous().view(Bs, -1, T, N * D)

        a = None
        out = self.proj_drop(self.proj_net(V))
        return out, a
