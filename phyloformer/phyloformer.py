"""The phyloformer module contains the Phyloformer network as well as functions to 
create and load instances of the network from disk
"""
from typing import Any, Dict, Optional

import skbio
import torch
import torch.nn as nn
from ete3 import Tree
from scipy.special import binom

from phyloformer.attentions import KernelAxialMultiAttention


class AttentionNet(nn.Module):
    """Phyloformer Network"""

    def __init__(
        self,
        n_blocks: int = 1,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        device: str = "cpu",
        n_seqs: int = 20,
        seq_len: int = 200,
        **kwargs
    ):
        """Initializes internal Module state

        Parameters
        ----------
        n_blocks : int, optional
            Number of blocks in transformer, by default 1
        n_heads : int, optional
            Number of heads in multi-head attention, by default 4
        h_dim : int, optional
            Hidden dimension, by default 64
        dropout : float, optional
            Droupout rate, by default 0.0
        device : str, optional
            Device for model ("cuda" or "cpu"), by default "cpu"
        n_seqs : int, optional
            Number of sequences in input alignments, by default 20
        seq_len : int, optional
            Length of sequences in input alignment, by default 200

        Returns
        -------
        AttentionNet
            Functional instance of AttentionNet for inference/fine-tuning

        Raises
        ------
        ValueError
            If h_dim is not divisible by n_heads
        """

        if h_dim % n_heads != 0:
            raise ValueError(
                "The embedding dimension (h_dim) must be divisible"
                "by the number of heads (n_heads)!"
            )

        super(AttentionNet, self).__init__()
        # Initialize variables
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device

        self._init_seq2pair(n_seqs, seq_len)

        # Initialize Module lists
        self.rowAttentions = nn.ModuleList()
        self.columnAttentions = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()

        layers_1_1 = [
            nn.Conv2d(in_channels=22, out_channels=h_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        ]
        self.block_1_1 = nn.Sequential(*layers_1_1)
        self.norm = nn.LayerNorm(h_dim)
        self.pwFNN = nn.Sequential(
            *[
                nn.Conv2d(in_channels=h_dim, out_channels=1, kernel_size=1, stride=1),
                nn.Dropout(dropout),
                nn.Softplus(),
            ]
        )
        for i in range(self.n_blocks):
            self.rowAttentions.append(
                KernelAxialMultiAttention(h_dim, n_heads, n=seq_len).to(device)
            )
            self.columnAttentions.append(
                KernelAxialMultiAttention(h_dim, n_heads, n=int(binom(n_seqs, 2))).to(
                    device
                )
            )
            self.layernorms.append(nn.LayerNorm(h_dim).to(device))
            self.fNNs.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_channels=h_dim,
                            out_channels=h_dim * 4,
                            kernel_size=1,
                            stride=1,
                            device=device,
                        ),
                        nn.Dropout(dropout),
                        nn.GELU(),
                        nn.Conv2d(
                            in_channels=h_dim * 4,
                            out_channels=h_dim,
                            kernel_size=1,
                            stride=1,
                            device=device,
                        ),
                    ],
                    nn.Dropout(dropout)
                )
            )

    def _init_seq2pair(self, n_seqs: int, seq_len: int):
        """Initialize Seq2Pair matrix"""
        self.n_seqs = n_seqs
        self.seq_len = seq_len
        self.n_pairs = int(binom(n_seqs, 2))

        seq2pair = torch.zeros(self.n_pairs, self.n_seqs)
        k = 0
        for i in range(self.n_seqs):
            for j in range(i + 1, self.n_seqs):
                seq2pair[k, i] = 1
                seq2pair[k, j] = 1
                k = k + 1

        self.seq2pair = seq2pair.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Doed a forward pass through the Phyloformer network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (shape 1\*22\*n_seqs\*seq_len)

        Returns
        -------
        torch.Tensor
            Output tensor (shape 1\*n_pairs)

        Raises
        ------
        ValueError
            If the tensors aren't the right shape
        """
        attentionmaps = []
        # 2D convolution that gives us the features in the third dimension
        # (i.e. initial embedding of each amino acid)
        out = self.block_1_1(x)
        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))  # pair representation

        # From here on the tensor has shape (batch_size,features,nb_pairs,seq_len), all
        # the transpose/permute allow to apply layernorm and attention over the desired
        # dimensions and are then followed by the inverse transposition/permutation
        # of dimensions

        out = self.norm(out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            att, a = self.rowAttentions[i](out.permute(0, 2, 3, 1))
            out = att.permute(0, 3, 1, 2) + out  # row attention+residual connection
            out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                -1, -3
            )  # layernorm

            # COLUMN ATTENTION
            att, a = self.columnAttentions[i](out.permute(0, 3, 2, 1))

            attentionmaps.append(a)
            out = att.permute(0, 3, 2, 1) + out  # column attention+residual connection
            out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                -1, -3
            )  # layernorm

            # FEEDFORWARD
            out = self.fNNs[i](out) + out
            if i != self.n_blocks - 1:
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm

        # After this last convolution we have (batch_size,1,nb_pairs,seq_len)
        out = self.pwFNN(out)
        # Averaging over positions and removing the extra dimensions
        # we finally get (batch_size,nb_pairs)
        out = torch.squeeze(torch.mean(out, dim=-1))

        return out

    def _get_architecture(self) -> Dict[str, Any]:
        """Returns architecture parameters of the model

        Returns
        -------
        Dict[str, Any]
            Dictionnary containing model architecture
        """
        return {
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "h_dim": self.h_dim,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "n_seqs": self.n_seqs,
            "device": self.device,
        }

    def save(self, path: str) -> None:
        """Saves the model parameters to disk

        Parameters
        ----------
        path : str
            Path to save the model to
        """
        torch.save(
            {
                "architecture": self._get_architecture(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    def infer_dm(
        self, X: torch.Tensor, ids: Optional[list[str]] = None
    ) -> skbio.DistanceMatrix:
        """Infers a phylogenetic distance matrix from embedded alignment tensor

        Parameters
        ----------
        X : torch.Tensor
            Input alignment, embedded as a tensor (shape 22\*n_seq\*seq_len)
        ids : list[str], optional
            Identifiers of the sequences in the input tensor, by default None

        Returns
        -------
        skbio.DistanceMatrix
            Phylolgenetic distance matrix inferred by Phyloformer

        Raises
        ------
        ValueError
            If the tensors aren't the right shape
        """

        # reshape from 22*n_seq*seq_len to 1*22*n_seq*seq_len
        tensor = X[None, :, :]
        tensor = tensor.to(self.device)

        # Infer distances
        with torch.no_grad():
            predictions = self(tensor.float())
        predictions = predictions.view(self.n_pairs)

        # Build distance matrix
        nn_dist = {}
        cursor = 0
        for i in range(self.n_seqs):
            for j in range(self.n_seqs):
                if i == j:
                    nn_dist[(i, j)] = 0
                if i < j:
                    pred = predictions[cursor].item()
                    pred = float("%.6f" % (pred))
                    nn_dist[(i, j)], nn_dist[(j, i)] = pred, pred
                    cursor += 1

        return skbio.DistanceMatrix(
            [[nn_dist[(i, j)] for j in range(self.n_seqs)] for i in range(self.n_seqs)],
            ids=ids,
        )

    def infer_tree(
        self,
        X: torch.Tensor,
        ids: Optional[list[str]] = None,
        dm: Optional[skbio.DistanceMatrix] = None,
    ) -> Tree:
        """Infers a phylogenetic tree from an embedded alignment tensor

        Parameters
        ----------
        X : torch.Tensor
            Input alignment, embedded as a tensor (shape 22\*n_seq\*seq_len)
        ids : list[str], optional
            Identifiers of the sequences in the input tensor, by default None
        dm : skbio.DistanceMatrix, optional
            Precomputed distance matrix if you have already run `AttentionNet.infer_dm`
            on your own, by default None

        Returns
        -------
        Tree
            Phylogenetic tree computed with neighbour joining from the distance matrix
            inferred by Phyloformer

        Raises
        ------
        ValueError
            If the tensors aren't the right shape
        """
        phyloformer_dm = dm if dm is not None else self.infer_dm(X, ids)
        nn_newick_str = skbio.tree.nj(phyloformer_dm, result_constructor=str)

        return Tree(nn_newick_str)


def _init_model(model: AttentionNet, state_dict: dict, single_gpu: bool):
    """Loads  a state_dict into a Phyloformer model

    Parameters
    ----------
    model : AttentionNet
        Phyloformer model to populate
    state_dict : dict
        State dict to populate the model with
    single_gpu: bool
        Wether inference/fine-tuning will be done on a single GPU
    """

    # Remove "module." from keys for models trained on multiple gpus
    new_state_dict = (
        {k.replace("module.", ""): v for k, v in state_dict.items()}
        if single_gpu
        else state_dict
    )

    model.load_state_dict(new_state_dict, strict=True)


def load_model(path: str, device: str = "cpu", single_gpu: bool = True) -> AttentionNet:
    """Load a Phyloformer istance froms disk

    Parameters
    ----------
    path : str
        Path to model saved with AttentionNet.save()
    device : str, optional
        Device to load model to ("cpu" or "cuda"), by default "cpu"
    single_gpu: bool, optional
        Wether inference/fine-tuning will be done on a single GPU, by default True

    Returns
    -------
    AttentionNet
        Functional instance of AttentionNet for inference/fine-tuning

    Raises
    ------
    ValueError
        If the file does not contain the state_dict and model architecture parameters
    """

    loaded = torch.load(path, map_location=device)
    if loaded.get("state_dict") is None or loaded.get("architecture") is None:
        raise ValueError(
            "Error loading model. Saved file must contain both a 'state_dict' "
            "and a 'architecture' entry"
        )

    model = AttentionNet(**loaded["architecture"], device=device)
    _init_model(model, loaded["state_dict"], single_gpu)
    model.to(device)

    return model


def load_state_dict(
    model_path: str,
    n_blocks: int = 6,
    h_dim: int = 64,
    n_heads: int = 4,
    n_seqs: int = 20,
    seq_len: int = 200,
    device: str = "cpu",
    dropout: float = 0.0,
    single_gpu=True,
    **kwargs
) -> AttentionNet:
    """This function loads a pre-trained phyloformer model to be used for inference
    or fine-tuning, from a file containing the state dict. The user must specify the
    model's arrchitecture parameters themselves. For models saved with
    `AttentionNet.save` you should use the `load_model` function instead.

    Parameters
    ----------
    model_path : str
        Path to the saved model (Either the saved state dict or a dictionnary containing
        the state dict)
    n_blocks : int, optional
        Number of attention blocks, by default 6
    h_dim : int, optional
        Hidden dimension of attention blocks, by default 64
    n_heads : int, optional
        Number of attentions heads per block, by default 4
    n_seqs : int, optional
        Number of sequences in input alignments, by default 20
    seq_len : int, optional
        Length of sequences in input alignments, by default 200
    device : str, optional
        PyTorch device, by default "cpu"
    dropout : float, optional
        Drop out rate, by default 0.0
    single_gpu: bool, optional
        Wether inference/fine-tuning will be done on a single GPU, by default True

    Returns
    -------
    AttentionNet
        A functional instance of AttentionNet ready for use for inference/fine-tuning
    """
    model = AttentionNet(
        n_blocks=n_blocks,
        n_heads=n_heads,
        h_dim=h_dim,
        n_seqs=n_seqs,
        seq_len=seq_len,
        dropout=dropout,
        device=device,
    )
    loaded = torch.load(model_path, map_location=device)
    state_dict = (
        loaded if loaded.get("state_dict") is None else loaded.get("state_dict")
    )
    _init_model(model, state_dict, single_gpu)
    model.to(device)

    return model
