"""
The data module contains functions that are used to load alignment data in a format that
the Phyloformer network understands. 
"""
import os
from typing import Tuple

import numpy as np
import skbio
import torch
from Bio import SeqIO
from torch.utils.data import Dataset

AMINO_ACIDS = np.array(list("ARNDCQEGHILKMFPSTWYVX-"))


class TensorDataset(Dataset):
    def __init__(self, directory: str):
        super(TensorDataset, self).__init__()
        self.directory = directory
        self.index = [
            filepath for filepath in os.listdir(self.directory)
            if filepath.endswith(".tensor")
        ]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, index: int):
        return torch.load(os.path.join(self.in_dir, "X" + self.t[index])), torch.load(
            os.path.join(self.in_dir, "y" + self.t[index])
        )


def load_alignment(path: str) -> Tuple[torch.Tensor, list[str]]:
    """Loads an alignment into a tensor digestible by the Phyloformer network

    Parameters
    ----------
    path : str
        Path to a fasta file containing the alignment

    Returns
    -------
    Tuple[torch.Tensor, list[str]]
        a tuple containing:
         - a tensor representing the alignment (shape 22 * seq_len * nb_seq)
         - a list of ids of the sequences in the alignment
    """

    tensor = []
    parsed = _parse_alignment(path)
    for sequence in parsed.values():
        tensor.append(
            torch.from_numpy(_sequence_to_one_hot(sequence)).t().view(22, 1, -1)
        )
    tensor = torch.cat(tensor, dim=1).transpose(-1, -2)

    return tensor, list(parsed.keys())


def load_dataset(path: str) -> list[torch.Tensor]:
    pass


def _parse_alignment(path: str) -> dict[str, str]:
    """Parser a fasta alignment

    Parameters
    ----------
    path : str
        Path to .fasta alignment file

    Returns
    -------
    dict[str,str]
        A dictionnary with ids as keys and sequence as values
    """
    return {record.id: str(record.seq) for record in SeqIO.parse(path, format="fasta")}


def _sequence_to_one_hot(seq: str) -> np.ndarray:
    """Encode an amino acid sequence with one-hot encoding

    Parameters
    ----------
    seq : str
        Sequence of amino acids to encode

    Returns
    -------
    np.ndarray
        Encoded sequence (shape 22\*seq_len)
    """
    return np.array([(AMINO_ACIDS == aa).astype(int) for aa in seq])


def write_dm(dm: skbio.DistanceMatrix, path: str):
    """Write a distance matrix to disk in the square Phylip matrix format

    Parameters
    ----------
    dm : skbio.DistanceMatrix
        Distance matrix to save
    path : str
        Path where to save the matrix
    """    

    with open(path, "w+") as file:
        file.write(f"{len(dm.ids)}\n")
        for id, dists in zip(dm.ids, dm.data):
            line = ' '.join(str(dist) for dist in dists)
            file.write(f"{id}     {line}\n")