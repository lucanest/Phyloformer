import argparse
import json
import os
from typing import Optional
from time import time
from pathlib import Path
from glob import glob

import torch  # type:ignore
from scipy.special import binom
from torch import nn
from tqdm import tqdm

ALPHABET = b"ARNDCQEGHILKMFPSTWYVX-"
LOOKUP = {char: index for index, char in enumerate(ALPHABET)}


def MRE_loss(*args, **kwargs):
    pass


def load_alignment(filepath):
    sequences, ids = [], []

    with open(filepath, "rb") as aln:
        for line in aln:
            line = line.strip()
            if line.startswith(b">"):
                ids.append(line[1:].decode("utf8"))
                sequences.append([])
            else:
                for char in line:
                    sequences[-1].append(LOOKUP[char])

    seqs = torch.tensor(sequences)
    seqs = torch.nn.functional.one_hot(seqs, num_classes=len(ALPHABET)).permute(2, 1, 0)

    return seqs, ids


def vec_to_phylip(preds, ids):
    n = len(ids)
    dm = torch.zeros((n, n)).type_as(preds)
    i = torch.triu_indices(row=n, col=n, offset=1)
    dm[i[0], i[1]] = preds

    s = f"{n}\n"
    for id, row in zip(ids, dm + dm.T):
        row_s = " ".join([f"{x:.10f}" for x in row])
        s += f"{id} {row_s}\n"

    return dm + dm.T, s


def get_batch_dms(batch_preds, n):
    dms = torch.zeros((batch_preds.shape[0], n, n)).type_as(batch_preds)
    i = torch.triu_indices(row=n, col=n, offset=1)
    dms[:, i[0], i[1]] = batch_preds

    return dms + dms.transpose(-1, -2)


def dm_to_phylip(dm, ids):
    s = f"{len(ids)}\n"
    for id, row in zip(ids, dm):
        row_s = " ".join([f"{x:.10f}" for x in row])
        s += f"{id} {row_s}\n"
    return s


# Removes all extensions (useful for still matching on predicted trees)
# def stem(path):
#     filename = pathlib.PurePath(path)
#     return str(filename.stem).removesuffix("".join(filename.suffixes))


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


# Large Seq2Pair bas matrix
SEQ2PAIR = seq2pair(200)


def adaptable_seq2pair(n_seqs: int, global_seq2pair):
    """Initialize Seq2Pair matrix"""
    max_n_seqs = global_seq2pair.shape[1]
    if n_seqs > max_n_seqs:
        raise ValueError(
            f"n_seqs must be smaller or equal to {max_n_seqs} "
            "(or pre-compute a larger global_seq2pair)"
        )
    # seq2pair = torch.clone(global_seq2pair)
    # Retain n_seqs columns (sequences) and rows (pairs) that only
    # involve these sequences. Arbitrarilly using the first
    # columns, but any subset of n_seqs columns would do.
    mask = (torch.norm(global_seq2pair[:, n_seqs:], dim=1) == 0).squeeze()
    seq2pair = global_seq2pair[mask, :n_seqs]
    del mask
    return seq2pair


class BaseAttention(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
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

        self.embed_dim = embed_dim
        self.qk_dim = qk_dim
        self.nb_heads = nb_heads
        self.dropout = dropout

        self.head_dim = embed_dim // nb_heads
        self.head_qk_dim = qk_dim // nb_heads

        self.k_proj = nn.Linear(embed_dim, qk_dim)
        self.q_proj = nn.Linear(embed_dim, qk_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.elu = nn.ELU()
        self.eps = eps

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)


class ScaledLinearAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(nb_heads, embed_dim, nb_heads, dropout, eps)

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


class AxialLinearTransformerLayer(nn.Module):
    """ESM-like axial transformer layer"""

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


class AxialLinearTransformer(nn.Module):
    """ESM-like implementation of phyloformer"""

    def __init__(
        self,
        nb_blocks: int = 6,
        nb_heads: int = 4,
        embed_dim: int = 64,
        dropout: float = 0.0,
        n_seqs: int = 20,
        seq_len: int = 200,
        normalize: bool = True,
        heterodims: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.nb_heads = nb_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.normalize = normalize
        self.heterodims = heterodims
        self.device = device

        self.n_seqs = n_seqs
        self.seq_len = seq_len

        self._init_seq2pair(20)

        self.embedding_block = nn.Sequential(
            nn.Conv2d(
                in_channels=22, out_channels=self.embed_dim, kernel_size=1, stride=1
            ),
            nn.ReLU(),
        )

        self.attention_blocks = nn.ModuleList(
            [
                AxialLinearTransformerLayer(
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
        if self.n_seqs == n_seqs:
            return self.seq2pair

    def _init_seq2pair(self, n_seqs: int):
        """Initialize Seq2Pair matrix"""

        self.n_seqs = n_seqs
        self.n_pairs = int(binom(n_seqs, 2))
        self.seq2pair = adaptable_seq2pair(n_seqs, SEQ2PAIR).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Infer evolutionnary distances with PhyloFormer")
    parser.add_argument("checkpoint", help="Path to model checkpoint to use")
    parser.add_argument("alignments", help="Path to alignment to infer tree for")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        required=False,
        help="Path to output distance matrix",
    )
    parser.add_argument("--trees", "-t", action="store_true", help="Output NJ trees")
    args = parser.parse_args()

    if args.trees:
        from skbio import DistanceMatrix
        from skbio.tree import nj

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Loading model Which means we don't need lightning anymore
    ckpt = torch.load(args.checkpoint, map_location=device)
    params = ckpt["hyper_parameters"]
    params["device"] = device
    model = AxialLinearTransformer(**params)
    model.load_state_dict(
        {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k != "model.seq2pair"
        },
        strict=False,
    )

    # Move model to correct place
    model = model.to(device)
    model.eval()

    # Path to dirs
    in_dir = os.path.abspath(args.alignments)
    out_dir = os.path.abspath(args.output_dir)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        prev_shape = None
        for alnpath in tqdm(glob(f"{in_dir}/*")):

            # Check if path has FASTA extension
            if not (
                alnpath.lower().endswith(".fa") or alnpath.lower().endswith(".fasta")
            ):
                raise ValueError(
                    "Input files must be fasta files (.fa or .fasta). Got " f"{alnpath}"
                )

            stem = Path(alnpath).stem
            matpath = os.path.join(out_dir, f"{stem}.phy")
            treepath = os.path.join(out_dir, f"{stem}.nj.nwk")

            aln, ids = load_alignment(alnpath)
            # Set seq2pair matrix
            if prev_shape is None or prev_shape != aln.shape[-1]:
                model._init_seq2pair(aln.shape[-1])
                prev_shape = aln.shape[-1]

            # Predict distance matrix
            preds = model(aln[None, :].to(device).float())

            # Write distance matrix to disk
            dm, phylip = vec_to_phylip(preds, ids)
            with open(matpath, "w") as outfile:
                outfile.write(phylip)

            # Write tree to disk if needed
            if args.trees:
                dm = DistanceMatrix(dm.cpu().detach().numpy(), ids=ids)
                with open(treepath, "w") as outfile:
                    outfile.write(nj(dm, result_constructor=str))
