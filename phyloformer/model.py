import torch  # type:ignore
from scipy.special import binom
from torch import nn

from .attention import ScaledLinearAttention


def seq2pair(n_seqs: int):
    """Initialize Seq2Pair matrix"""
    n_pairs = int(binom(n_seqs, 2))
    seq2pair = torch.zeros(n_pairs, n_seqs)
    k = 0
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            seq2pair[k, i] = 1
            seq2pair[k, j] = 1
            k = k + 1
    return seq2pair


def adaptable_seq2pair(n_seqs: int, global_seq2pair):
    """Initialize Seq2Pair matrix"""
    max_n_seqs = global_seq2pair.shape[1]
    if n_seqs > max_n_seqs:
        raise ValueError(
            f"n_seqs must be smaller or equal to {max_n_seqs} "
            "(or pre-compute a larger global_seq2pair)"
        )
    # Retain n_seqs columns (sequences) and rows (pairs) that only
    # involve these sequences. Arbitrarilly using the first
    # columns, but any subset of n_seqs columns would do.
    mask = (torch.norm(global_seq2pair[:, n_seqs:], dim=1) == 0).squeeze()
    seq2pair = global_seq2pair[mask, :n_seqs]
    del mask
    return seq2pair


# Global instance of a large Seq2Pair matrix
SEQ2PAIR = seq2pair(200)


class PhyloformerLayer(nn.Module):
    """Phyloformer's Transformer Layer"""

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float,
        normalize: bool = True,
        heterodims: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.dropout = dropout
        self.normalize = normalize
        self.heterodims = heterodims

        self.row_attention = ScaledLinearAttention(self.embed_dim, self.nb_heads)
        self.col_attention = ScaledLinearAttention(self.embed_dim, self.nb_heads)

        # Normalization layers
        self.row_norm = nn.LayerNorm(self.embed_dim)
        self.col_norm = nn.LayerNorm(self.embed_dim)
        self.ffn_norm = nn.LayerNorm(self.embed_dim)

        # Feed forward NN
        self.ffn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim * 4,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.embed_dim * 4,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(self, input):
        # Row attention sub-block
        res_row = input
        out = self.row_norm(input.transpose(-1, -3)).transpose(-1, -3)
        out = self.row_attention(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out + res_row  # residual connection

        # Col attention sub-block
        res_col = out
        out = self.col_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.col_attention(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        out = out + res_col

        # FFN sub-block
        res_ffn = out
        out = self.ffn_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.ffn(out)
        out = out + res_ffn

        return out


class Phyloformer(nn.Module):
    """Model architecture for Phyloformer"""

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        n_seqs: int = 20,
        seq_len: int = 200,
        normalize: bool = True,
        heterodims: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.nb_blocks = n_blocks
        self.nb_heads = n_heads
        self.embed_dim = h_dim
        self.dropout = dropout
        self.normalize = normalize
        self.heterodims = heterodims

        self.n_seqs = n_seqs
        self.seq_len = seq_len

        # Initialize seq2pair matrix
        self.seq2pair = adaptable_seq2pair(20, SEQ2PAIR)

        self.embedding_block = nn.Sequential(
            nn.Conv2d(
                in_channels=22, out_channels=self.embed_dim, kernel_size=1, stride=1
            ),
            nn.ReLU(),
        )

        self.attention_blocks = nn.ModuleList(
            [
                PhyloformerLayer(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                    normalize=self.normalize,
                    heterodims=self.heterodims,
                )
                for _ in range(self.nb_blocks)
            ]
        )

        self.pwFNN = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=1, kernel_size=1, stride=1
            ),
            nn.Dropout(self.dropout),
            nn.Softplus(),
        )

    def forward(self, input):
        # input: (batch_size, 22, seq_len, n_seqs)

        # Set seq2pair matrix if needed
        self._set_seq2pair(input.shape[-1])

        # Embed alignment to embed_dim
        out = self.embedding_block(input)
        # Pair representation -> (batch_size, embed_dim, nb_pairs, seq_len)
        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))

        # Attention
        for block in self.attention_blocks:
            out = block(out)

        # Convolution -> (batch_size, 1, nb_pairs, seq_len)
        out = self.pwFNN(out)

        # Average of sequence length -> (batch_size, nb_pairs)
        out = torch.squeeze(torch.mean(out, dim=-1))

        return out

    def _set_seq2pair(self, n_seqs: int):
        """Initialize Seq2Pair matrix"""

        # Don't do anything if the alignment shape is the same
        if self.n_seqs == n_seqs:
            return

        self.n_seqs = n_seqs
        self.n_pairs = int(binom(n_seqs, 2))

        # Generate new
        device = self.seq2pair.device
        self.seq2pair = adaptable_seq2pair(n_seqs, SEQ2PAIR).to(device)
