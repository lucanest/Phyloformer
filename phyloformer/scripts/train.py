import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
from time import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from phyloformer.data import TensorDataset
from phyloformer.phyloformer import AttentionNet
from phyloformer.training import init_training, load_checkpoint, training_loop

TITLE = """\
------------------------------------------------------------
______ _           _        __
| ___ \ |         | |      / _|
| |_/ / |__  _   _| | ___ | |_ ___  _ __ _ __ ___   ___ _ __
|  __/| '_ \| | | | |/ _ \|  _/ _ \| '__| '_ ` _ \ / _ \ '__|
| |   | | | | |_| | | (_) | || (_) | |  | | | | | |  __/ |
\_|   |_| |_|\__, |_|\___/|_| \___/|_|  |_| |_| |_|\___|_|
              __/ |
             |___/

------------------------------------------------------------"""


def init_loggers(log_option: Optional[str], identifier: str, logfile: str):
    log_tb, log_to_file = False, False
    if log_option == "tensorboard":
        log_tb = True
    elif log_option == "both":
        log_tb, log_to_file = True, True
    elif log_option is not None:
        log_to_file = True

    writer = None
    if log_tb:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(comment=identifier)

    log_file = logfile if log_to_file else None

    return writer, log_file


def main():
    parser = argparse.ArgumentParser(description="Train a phyloformer model")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="/path/ to input directory containing the\
    the tensor pairs on which the model will be trained",
    )
    parser.add_argument(
        "-v",
        "--validation",
        required=False,
        type=str,
        help="/path/ to input directory containing the\
    the tensor pairs on which the model will be evaluated. If left empty \
    10%% of the training set will be used as validation data.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="/path/ to the configuration json file for the hyperparameters",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=".",
        type=str,
        help="/path/ to output directory where the model parameters\
        and the metrics will be saved (default: current directory)",
    )
    parser.add_argument(
        "-l",
        "--load",
        required=False,
        type=str,
        help="Load training checkpoint",
        metavar="CHECKPOINT",
    )
    parser.add_argument(
        "-g",
        "--log",
        required=False,
        choices=["tensorboard", "file", "both"],
        default="file",
        type=str,
        help="How to log training process",
    )
    parser.add_argument(
        "--logfile",
        required=False,
        type=str,
        default=f"training-{datetime.now()}.log",
        help="path to save log at",
    )
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    print(TITLE)

    print("\nCalling training script with following arguments:")
    pprint(args.__dict__)
    print()

    print("Training config: ")
    pprint(config)
    print()

    device = "cpu"
    if config.get("device") == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif config.get("device") == "mps" and torch.backends.mps.is_available():
        device = "mps"

    identifier = (
        f"LR_{config['learning_rate']}_O_{config['optimizer']}_"
        f"L_{config['loss']}_E_{config['epochs']}_BS_{config['batch_size']}"
        f"NB_{config['n_blocks']}_NH_{config['n_heads']}_HD_{config['h_dim']}_"
        f"{'A_' if config['amp'] else ''}D_{config['dropout']}"
    )

    start_time = time()

    # Set seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tb_writer, log_file = init_loggers(args.log, identifier, args.logfile)

    print("Loading model, scheduler and optimizer.")
    if args.load is not None:
        print(f"Loading from checkpoint: {args.load}")
        model, optimizer, scheduler, criterion, _ = load_checkpoint(
            args.load, device=device
        )
    else:
        model = AttentionNet(**config)
        model.to(device)
        optimizer, scheduler, criterion = init_training(model, **config)

    print("Loading training and validation data.")
    if args.validation is not None:
        train_data = DataLoader(
            TensorDataset(args.input), batch_size=config["batch_size"]
        )
        val_data = DataLoader(
            TensorDataset(args.validation), batch_size=config["batch_size"]
        )
    else:
        tensor_files = list(os.listdir(args.input))
        n_tensors = int(len(tensor_files) * 0.1)
        sampled_indices = set(
            random.Random(seed).sample(range(len(tensor_files)), n_tensors)
        )
        train_ids, val_ids = [], []
        for i, file in enumerate(tensor_files):
            if i in sampled_indices:
                val_ids.append(file)
            else:
                train_ids.append(file)
        train_data = DataLoader(
            TensorDataset(args.input, filter=train_ids), batch_size=config["batch_size"]
        )
        val_data = DataLoader(
            TensorDataset(args.input, filter=val_ids), batch_size=config["batch_size"]
        )
    train_len = len(train_data.dataset.pairs)
    val_len = len(val_data.dataset.pairs)
    print(f"Model will train on {train_len} training and {val_len} validation tensors.")
    print()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    best_model, epoch = training_loop(
        model,
        optimizer,
        scheduler,
        criterion,
        train_data,
        val_data,
        epochs=config["epochs"],
        config=config,
        log_file=log_file,
        writer=tb_writer,
        device=device,
        checkpoint_path=os.path.join(args.output, f"{identifier}.checkpoint.pt"),
        best_path=os.path.join(args.output, f"{identifier}.best_checkpoint.pt"),
    )

    print(f"\nBest model gotten after {epoch} epochs!")
    best_model.save(os.path.join(args.output, f"{identifier}.best_model.pt"))

    # Cleanup
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    print(f"total elapsed time: {time()-start_time} seconds")


if __name__ == "__main__":
    main()
