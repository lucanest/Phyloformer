"""The training submodule contains necessary functions to train a Phyloformer network
"""


from typing import Any, Dict, Tuple, Union

import torch

from phyloformer.phyloformer import AttentionNet


Scheduler = Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]

LOSSES = {
    "L1": torch.nn.L1Loss(),
    "L2": torch.nn.MSELoss(),
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}


def init_training(
    model: AttentionNet,
    optimizer: str = "Adam",
    learning_rate: float = 1e-5,
    loss: str = "L2",
    warmup: bool = False,
    warmup_steps: int = 20,
    **kwargs,
) -> Tuple[torch.optim.Optimizer, Scheduler, torch.nn.modules.loss._Loss,]:
    """Creates and returns an optimizer, learning rate scheduler and loss function
    from a config dict

    Parameters
    ----------
    model : AttentionNet
        Phyloformer model to train
    optimizer : str, optional
        Optimization algorithm ("Adam", "AdamW", or "SGD"), by default "Adam"
    learning_rate : float, optional
        Learning rate to apply, by default 1e-5
    loss : str, optional
        Loss function (choices "L1" or "L2"), by default "L2"
    warmup : bool, optional
        Wether to do learning rate warmupt, by default False
    warmup_steps : int, optional
        Number of warmup steps, by default 20

    Returns
    -------
    Tuple[torch.optim.Optimizer,Scheduler,torch.nn.modules.loss._Loss]
        A tuple containing:
         - The optimizer
         - The learning rate scheduler
         - The loss function
    """
    optimizer_instance = _init_optimizer(
        model=model, algorithm=optimizer, learning_rate=learning_rate
    )
    scheduler = _init_scheduler(
        optimizer=optimizer_instance, warmup=warmup, warmup_steps=warmup_steps
    )
    criterion = _init_loss(loss=loss.upper())

    return optimizer_instance, scheduler, criterion


def training_loop():
    """Trains the Phyloformer model on a dataset with given optimizer and scheduler"""
    pass


def save_checkpoint(
    model: AttentionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    config: Dict[str, Any],
    path: str,
):
    """Saves a checkpoint of the current training process

    Parameters
    ----------
    model : AttentionNet
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer to save
    scheduler : Scheduler
        Learning rate scheduler to save
    config : Dict[str, Any]
        Config with which training was started
    path : str
        Path to save the checkpoint to
    """
    checkpoint = {
        "model": {
            "architecture": model._get_architecture(),
            "state_dict": model.state_dict(),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
) -> Tuple[
    AttentionNet,
    torch.optim.Optimizer,
    Scheduler,
    torch.nn.modules.loss._Loss,
    Dict[str, Any],
]:
    """Loads a model training checkpoint from disk

    Parameters
    ----------
    path : str
        Path to the saved checkpoint

    Returns
    -------
    Tuple[\
    AttentionNet, torch.optim.Optimizer, Scheduler, \
    torch.nn.modules.loss._Loss, Dict[str, Any]]
        A tuple containing:
         - The model
         - The optimizer
         - The learning rate scheduler
         - The loss function
         - The training configuration
    """

    checkpoint = torch.load(path)
    config = checkpoint["config"]

    # Loading model
    model = AttentionNet(**checkpoint["model"]["architecture"])
    model.load_state_dict(checkpoint["model"]["state_dict"], strict=True)

    # Loading other objects
    optimizer, scheduler, loss = init_training(model, **config)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, scheduler, loss, config


def _init_optimizer(
    algorithm: str, model: AttentionNet, learning_rate: float
) -> torch.optim.Optimizer:
    """Instanciates the optimizer to train a Phyloformer model

    Parameters
    ----------
    algorithm : str
        Which optimization algorithm to use ('Adam', 'AdamW', or 'SGD')
    model : AttentionNet
        Phyloformer model you want to train
    learning_rate : float
        Learning rate to apply

    Returns
    -------
    torch.optim.Optimizer
        Optimizer to use in model training

    Raises
    ------
    ValueError
        If the specified optimization algorithm is not in the available options
    """
    if algorithm not in OPTIMIZERS:
        raise ValueError(
            f"The optimizer algorithm must be one of the following: {OPTIMIZERS.keys()}"
        )

    return OPTIMIZERS[algorithm](model.parameters(), lr=learning_rate)


def _init_scheduler(
    optimizer: torch.optim.Optimizer, warmup: bool, warmup_steps: int
) -> Scheduler:
    """Instanciates a learning rate scheduler for Phyloformer model training

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer that will be used during training
    warmup : bool
        Wether to do a learning rate warmup
    warmup_steps : int
        Number of warmup steps

    Returns
    -------
    Scheduler
        The learning rate scheduler instance
    """
    if warmup:
        print("Warmup is not implemented yet, instanciating LR-Plateau scheduler")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True
    )


def _init_loss(loss: str) -> torch.nn.modules.loss._Loss:
    """Instanciates a loss function object

    Parameters
    ----------
    loss : str
        The name of the loss function to use ("L1" or "L2")

    Returns
    -------
    torch.nn.modules.loss._Loss
        The loss function that will be used during training

    Raises
    ------
    ValueError
        If the specified loss is not available
    """
    if loss not in LOSSES:
        raise ValueError(
            f"The loss function must be one of the following: {LOSSES.keys()}"
        )
    return LOSSES[loss]
