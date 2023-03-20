"""
The metrics submodule contains functions to compute useful metrics and measures for 
the phyloformer project
"""

import ete3
import torch


def MAE(input: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the Mean Absolute Error

    Parameters
    ----------
    input : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target values

    Returns
    -------
    float
        Mean Absolute Error
    """
    return torch.nn.L1Loss(input, target).item()


def MRE(input: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the Mean Relative Error

    Parameters
    ----------
    input : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target values

    Returns
    -------
    float
        Mean Relative Error
    """
    return torch.mean(torch.abs(input - target) / target).item()


def RF(tree1: ete3.Tree, tree2: ete3.Tree) -> float:
    """Computes the Robinson-Foulds distance between two phylogenetic trees

    Parameters
    ----------
    tree1 : ete3.Tree
        First tree
    tree2 : ete3.Tree
        Second tree

    Returns
    -------
    float
        Robinson-Foulds Distance
    """
    return tree1.compare(tree2, unrooted=True)["norm_rf"]
