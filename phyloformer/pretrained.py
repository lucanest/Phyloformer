"""The pretrained module contains functions to load functional pre-trainde instances 
of the Phyloformer model.
"""
import os

import torch

from phyloformer.phyloformer import AttentionNet

ROOT = os.path.realpath(os.path.dirname(__file__))


def seqgen(device: str = "cpu") -> AttentionNet:
    """Loads a pre-trained Phyloformer model.
    This model was trained on Seq-Gen simulations

    Parameters
    ----------
    device : str, optional
        Device to load model to ("cpu" or "cuda"), by default "cpu"

    Returns
    -------
    AttentionNet
        A pre-trained instance of Phyloformer
    """
    model = AttentionNet(n_blocks=6, device=device)
    state_dict = torch.load(
        os.path.join(ROOT, "pretrained_models", "seqgen_model_state_dict.pt"),
        map_location=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    return model


def evosimz(device: str = "cpu") -> AttentionNet:
    """Loads a pre-trained Phyloformer model.
    This model was trained on Evosimz simulations

    Parameters
    ----------
    device : str, optional
        Device to load model to ("cpu" or "cuda"), by default "cpu"

    Returns
    -------
    AttentionNet
        A pre-trained instance of Phyloformer
    """
    model = AttentionNet(n_blocks=6, device=device)
    state_dict = torch.load(
        os.path.join(ROOT, "pretrained_models", "evosimz_model_state_dict.pt"),
        map_location=device
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    return model
