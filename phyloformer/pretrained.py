import os

import torch

from phyloformer.phyloformer import AttentionNet

ROOT = os.path.realpath(os.path.dirname(__file__))


def seqgen() -> AttentionNet:
    """Loads a pre-trained Phyloformer model.
    This model was trained on Seq-Gen simulations

    Returns
    -------
    AttentionNet
        A pre-trained instance of Phyloformer
    """
    model = AttentionNet(n_blocks=6)
    state_dict = torch.load(
        os.path.join(ROOT, "pretrained_models", "seqgen_model_state_dict.pt")
    )
    model.load_state_dict(state_dict, strict=True)

    return model


def evosimz() -> AttentionNet:
    """Loads a pre-trained Phyloformer model.
    This model was trained on Evosimz simulations

    Returns
    -------
    AttentionNet
        A pre-trained instance of Phyloformer
    """
    model = AttentionNet(n_blocks=6)
    state_dict = torch.load(
        os.path.join(ROOT, "pretrained_models", "evosimz_model_state_dict.pt")
    )
    model.load_state_dict(state_dict, strict=True)

    return model
