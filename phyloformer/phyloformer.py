import torch
import torch.nn as nn
from scipy.special import binom

from phyloformer.attentions import KernelAxialMultiAttention


class AttentionNet(nn.Module):
    """Phyloformer Network"""

    def __init__(
        self,
        n_blocks: int =1,
        n_heads: int =4,
        h_dim: int =64,
        dropout: float =0.0,
        device: str ="cpu",
        n_seq: int =20,
        seq_len: int =200,
    ) :
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
        n_seq : int, optional
            Number of sequences in input alignments, by default 20
        seq_len : int, optional
            Length of sequences in input alignment, by default 200

        Returns
        -------
        AttentionNet
            Functional instance of AttentionNet for inference/fine-tuning
        """        
        super(AttentionNet, self).__init__()
        # Initialize variables
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.h_dim = h_dim
        self.dropout = dropout
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.device = device

        self.n_pairs = int(binom(n_seq, 2))
        self._init_seq2pair()

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
                KernelAxialMultiAttention(h_dim, n_heads, n=int(binom(n_seq, 2))).to(
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

    def _init_seq2pair(self):
        """Initialize Seq2Pair matrix"""
        seq2pair = torch.zeros(self.n_pairs, self.n_seq)
        k = 0
        for i in range(self.n_seq):
            for j in range(i + 1, self.n_seq):
                seq2pair[k, i] = 1
                seq2pair[k, j] = 1
                k = k + 1

        self.seq2pair = seq2pair.to(self.device)

    def forward(self, x):
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
        # return out, attentionmaps

    def save(self, path: str) -> None:
        """Saves the model parameters to disk

        Parameters
        ----------
        path : str
            Path to save the model to
        """
        architecture = {
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "h_dim": self.h_dim,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "n_seq": self.n_seq,
            "device": self.device,
        }
        torch.save(
            {"architecture": architecture, "state_dict": self.state_dict()}, path
        )


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


def load_model(path: str, single_gpu: bool = True) -> AttentionNet:
    """Load a Phyloformer istance froms disk

    Parameters
    ----------
    path : str
        Path to model saved with AttentionNet.save()
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

    loaded = torch.load(path)
    if loaded.get("state_dict") is None or loaded.get("architecture") is None:
        raise ValueError(
            "Error loading model. Saved file must contain both a 'state_dict' "
            "and a 'architecture' entry"
        )

    model = AttentionNet(**loaded["architecture"])
    _init_model(model, loaded["state_dict"], single_gpu)

    return model


def load_state_dict(
    model_path: str,
    n_blocks: int = 6,
    h_dim: int = 64,
    n_heads: int = 4,
    n_seq: int = 20,
    seq_len: int = 200,
    device: str = "cpu",
    dropout: float = 0.0,
    single_gpu=True,
    **kwargs
) -> AttentionNet:
    """This function loads a pre-trained phyloformer model to be used for inference
    or fine-tuning, from a file containing the state dict. The user must specify the
    model's arrchitecture parameters themselves. For models saved with 
    AttentionNet.save() you should use the phyloformer.phyloformer.load_model() 
    function instead.

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
    n_seq : int, optional
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
        n_seq=n_seq,
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
