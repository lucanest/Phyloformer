#!/usr/bin/env python3

import argparse
import os
import pathlib
import random
import re
import sys
import math
from itertools import combinations
from pprint import pprint
from typing import Optional

import dendropy
import lightning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
import lightning.pytorch.loggers as log
import torch  # type:ignore
from scipy.special import binom
from torch import nn

if torch.backends.mps.is_available():  # type:ignore
    pass

from torch.optim import Adam  # type:ignore
from torch.utils.data import DataLoader, Dataset  # type:ignore
from transformers import get_linear_schedule_with_warmup  # type:ignore


def MAE(input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool) -> float:
    """Computes the Mean Absolute Error"""

    if sqrt_preds:
        input = input**2
    return torch.nn.L1Loss()(input, target).detach()


def MRE(input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool) -> float:
    """Computes the Mean Relative Error"""
    if sqrt_preds:
        input = input**2
    return torch.mean(torch.abs(input - target) / target).detach()


ALPHABET = "ARNDCQEGHILKMFPSTWYVX-"
LOOKUP = {char: index for index, char in enumerate(ALPHABET)}


def load_alignment(filepath):
    sequences, ids = [], []

    with open(filepath, "r") as aln:
        for line in aln:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:])
                sequences.append([])
            else:
                for char in line:
                    sequences[-1].append(LOOKUP[char])

    seqs = torch.tensor(sequences)
    seqs = torch.nn.functional.one_hot(seqs, num_classes=len(ALPHABET)).permute(2, 1, 0)

    return seqs, ids


def load_distance_matrix(filepath, ids):
    distances = []

    with open(filepath, "r") as treefile:
        tree = dendropy.Tree.get(file=treefile, schema="newick")
    taxa = tree.taxon_namespace
    dm = tree.phylogenetic_distance_matrix()
    for tip1, tip2 in combinations(ids, 2):
        l1, l2 = taxa.get_taxon(tip1), taxa.get_taxon(tip2)
        distances.append(dm.distance(l1, l2))

    return torch.tensor(distances)


def listdir_paths(root):
    return [os.path.join(root, file) for file in os.listdir(root)]


# Removes all extensions (useful for still matching on predicted trees)
def stem(path):
    filename = pathlib.PurePath(path)
    return str(filename.stem).removesuffix("".join(filename.suffixes))


def make_pairs(treefiles, alnfiles, regex):
    alndict = {stem(alnfile): alnfile for alnfile in alnfiles}
    pairs = []
    for treefile in treefiles:
        if not (treefile.endswith(".nwk") or treefile.endswith(".newick")):
            continue
        if regex is not None and not regex.search(treefile):
            continue
        alnfile = alndict.get(stem(treefile))
        if alnfile is None:
            print(f"Tree: {treefile}")
            print(f"Tree stem: {stem(treefile)}")
            raise IndexError(f"Tree: {treefile} has no corresponding alignment.")
        pairs.append((treefile, alnfile))
    return pairs


class PhyloDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        treefile, alnfile = self.pairs[index]
        x, ids = load_alignment(alnfile)
        y = load_distance_matrix(treefile, ids)

        return x, y


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


SEQ2PAIR = seq2pair(20)


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

        self.register_buffer("seq2pair", self._init_seq2pair(20))

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

        mat = adaptable_seq2pair(n_seqs, SEQ2PAIR)

        return mat


def choose_data(
    train_alignments, train_trees, train_regex, val_alignments, val_trees, val_regex
):
    # Choose training and validation examples
    if val_alignments is None and val_trees is None:
        regex = re.compile(train_regex) if train_regex is not None else None
        pairs = make_pairs(
            listdir_paths(train_trees), listdir_paths(train_alignments), regex
        )
        # Split data
        val_index = int(len(pairs) * 0.1)
        random.shuffle(pairs)
        val_pairs = pairs[:val_index]
        train_pairs = pairs[val_index:]
    elif val_alignments is not None and val_trees is not None:
        train_regex = re.compile(train_regex) if train_regex is not None else None
        val_regex = re.compile(val_regex) if val_regex is not None else None
        train_pairs = make_pairs(
            listdir_paths(train_trees),
            listdir_paths(train_alignments),
            train_regex,
        )
        val_pairs = make_pairs(
            listdir_paths(val_trees), listdir_paths(val_alignments), val_regex
        )
    else:
        raise ValueError(
            "You must either specify both validation trees and alignments "
            "or none of them."
        )

    return train_pairs, val_pairs


class LightningAxialTransformer(lightning.LightningModule):
    def __init__(
        self,
        nb_blocks: int,
        nb_heads: int,
        embed_dim: int,
        dropout: float,
        learning_rate: float,
        warmup_steps: int,
        total_steps: int,
        batch_size: int,
        optim_func,
        criterion,
    ):
        super().__init__()

        # Initialize model
        self.model = AxialLinearTransformer(
            n_blocks=nb_blocks,
            n_heads=nb_heads,
            h_dim=embed_dim,
            dropout=dropout,
        )

        # Optimizer stuff
        self.optim_func = optim_func
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.criterion = criterion
        self.batch_size = batch_size

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        loss = self.criterion(y_hat, y.type_as(y_hat).squeeze())
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        y = y.type_as(y_hat).squeeze()

        # Compute validation metrics and log them
        loss = self.criterion(y_hat, y)
        d = {"val_mre": MRE(y_hat, y, False), "val_mae": MAE(y_hat, y, False)}
        self.log_dict(dict(val_loss=loss, **d), sync_dist=True)

        return dict(loss=loss, **d)


class PhyloDataModule(lightning.LightningDataModule):
    def __init__(self, train_pairs, val_pairs, batch_size):
        super().__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(self.train_pairs),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS_TRAIN,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(self.val_pairs),
            batch_size=self.batch_size,
            num_workers=WORKERS_VAL,
            pin_memory=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "train PF instance", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train-trees", "-t", required=True, help="Directory with training trees"
    )
    parser.add_argument(
        "--train-alignments",
        "-a",
        required=True,
        help="Directory with training alignments",
    )
    parser.add_argument(
        "--val-trees", "-T", required=False, help="Directory with validation trees"
    )
    parser.add_argument(
        "--val-alignments",
        "-A",
        required=False,
        help="Directory with validation alignments",
    )
    parser.add_argument(
        "--dropout", "-D", default=0.0, type=float, help="Dropout proportion"
    )
    parser.add_argument(
        "--nb-blocks", "-b", default=6, type=int, help="Number of PF blocks"
    )
    parser.add_argument(
        "--embed-dim", "-d", default=64, type=int, help="Number of embedding dimensions"
    )
    parser.add_argument(
        "--nb-heads", "-H", default=4, type=int, help="Number of attention heads"
    )
    parser.add_argument(
        "--nb-epochs", "-e", default=100, type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--max-steps", "-m", default=None, type=int, help="Max number of training steps"
    )
    parser.add_argument(
        "--warmup-steps", "-w", default=5000, type=int, help="Number of warmup steps"
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        default=1e-4,
        type=float,
        help="Traget starting learning rate",
    )
    parser.add_argument(
        "--batch-size", "-s", default=4, type=int, help="Training batch size"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory to save losses and checkpoints",
    )
    parser.add_argument(
        "--train-regex", "-r", default=None, help="Regex to filter training examples"
    )
    parser.add_argument(
        "--val-regex", "-R", default=None, help="Regex to filter validation examples"
    )
    parser.add_argument(
        "--load-checkpoint", "-c", default=None, help="Path to training checkpoint"
    )
    parser.add_argument(
        "--check-val-every",
        "-C",
        default=10_000,
        type=int,
        help="Check validation dataset every n steps",
    )
    parser.add_argument(
        "--log-every",
        "-E",
        default=100,
        type=int,
        help="Log training loss every n steps",
    )
    parser.add_argument(
        "--no-improvement-stop",
        "-n",
        default=5,
        type=int,
        help="Number of checks with no improvement before stopping early",
    )
    parser.add_argument(
        "--hard-loss-ceiling",
        "-L",
        default=3.0,
        type=float,
        help="Max value of loss over which the training stops",
    )
    parser.add_argument(
        "--find-batch-size",
        "-f",
        action="store_true",
        help="Run the lightning batch_size finder (skips training)",
    )
    parser.add_argument(
        "--project-name",
        "-p",
        required=False,
        default="PHYLOFORMER_EXPERIMENTS",
        help="Project in which to save this run on WandB",
    )
    parser.add_argument(
        "--run-name",
        "-N",
        required=False,
        default=None,
        help="Name to give to the run on WandB",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiler for a few steps and exit"
    )

    args = parser.parse_args()

    # Initialize logger
    wandb_logger = log.WandbLogger(
        save_dir=args.output_dir,
        project=args.project_name,
        name=args.run_name,
        offline=True,
    )

    print(f"Training with args:\n{args}")

    VAL_CHECK_STEPS = args.check_val_every
    LOGGING_STEPS = args.log_every

    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    NUM_WORKERS = N_CPUS // 2

    global WORKERS_TRAIN
    global WORKERS_VAL
    WORKERS_TRAIN = max(NUM_WORKERS, N_CPUS - NUM_WORKERS)
    WORKERS_VAL = min(NUM_WORKERS, N_CPUS - NUM_WORKERS)

    print(f"Assigning {WORKERS_TRAIN} training, and {WORKERS_VAL} validation data-loading workers")

    # Make output directory for logging and checkpoints
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    seed = 1337
    lightning.pytorch.seed_everything(seed, workers=True)

    bs = args.batch_size
    hd = args.embed_dim
    nb = args.nb_blocks
    lr = args.learning_rate
    wp = args.warmup_steps
    dr = args.dropout

    train_pairs, val_pairs = choose_data(
        args.train_alignments,
        args.train_trees,
        args.train_regex,
        args.val_alignments,
        args.val_trees,
        args.val_regex,
    )

    # Check if we are on SLURM and grab env variables
    slurm_args = dict()
    if os.environ.get("SLURM_NODELIST") is not None:
        # Add SLURM arguments for distributed training
        slurm_args = {
            "accelerator": "gpu",
            "devices": int(os.environ["SLURM_GPUS_ON_NODE"]),
            "num_nodes": int(os.environ["SLURM_NNODES"]),
            "strategy": "ddp",
        }

    datamodule = PhyloDataModule(train_pairs, val_pairs, args.batch_size)
    n_gpus = slurm_args.get("devices", 1)
    total_steps = (
        math.ceil(len(train_pairs) / (args.batch_size * n_gpus)) * args.nb_epochs
    )

    criterion = torch.nn.L1Loss()
    model = LightningAxialTransformer(
        nb_blocks=args.nb_blocks,
        nb_heads=args.nb_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        batch_size=args.batch_size,
        optim_func=Adam,
        criterion=criterion,
    )

    # Find batch size and exit if necessary
    if args.find_batch_size:
        trainer = lightning.Trainer()
        tuner = Tuner(trainer)
        bs = tuner.scale_batch_size(
            model,
            mode="binsearch",
        )
        print(f"Lightning found an optimal batch size of: {bs}.")
        sys.exit(0)

    identifier = (
        f"LR_{args.learning_rate}_O_Adam_"
        f"L_L1_E_{args.nb_epochs}_BS_{args.batch_size}_"
        f"NB_{args.nb_blocks}_NH_{args.nb_heads}_HD_{args.embed_dim}_"
        f"D_{0.0}_W{args.warmup_steps}"
    )

       # Manually save hyperparameter string just in case
    wandb_logger.log_hyperparams({"identifier": identifier})

    # Early stopping callbacks if needed
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, f"checkpoints_{identifier}"),
            filename='{epoch}-{val_loss:.4f}-{train_loss:.4f}',
            save_top_k=-1, # Keep all checkpoints
            save_last=True, # Add symbolic link to point to last checkpoint
            every_n_train_steps=VAL_CHECK_STEPS,
            save_on_train_epoch_end=False, # Save after validation so the value is correct in filename
        )
    ]
    if args.hard_loss_ceiling is not None:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                check_finite=True,
                verbose=True,
                patience=10_000, # Simulate infinite patience so that this only checks for divergence
                divergence_threshold=args.hard_loss_ceiling,
            )
        )
    if args.no_improvement_stop is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.no_improvement_stop,
                verbose=True,
            )
        )

    trainer_args = {
        "max_epochs": args.nb_epochs,
        "log_every_n_steps": LOGGING_STEPS,
        "val_check_interval": VAL_CHECK_STEPS,
        "logger": wandb_logger,
        "callbacks": callbacks,
        **slurm_args,
    }

    # Run profiler for 30 steps
    if args.profile:
        trainer_args["max_steps"] = 10
        # trainer_args["profiler"] = "simple"
        trainer_args["profiler"] = PyTorchProfiler(
            dirpath=os.path.join(args.output_dir, f"profile_{identifier}"),
            profile_memory=True,
            record_shapes=True,
            with_modules=True,
        )


    print("INIT TRAINER WITH ARGS:")
    pprint(trainer_args)

    # Initialize trainer
    trainer = lightning.Trainer(**trainer_args)

    # Get training arguments
    train_args = dict(model=model)
    if args.load_checkpoint is not None:
        # train_args["model"] = LightningAxialTransformer.load_from_checkpoint(args.load_checkpoint)
        train_args["ckpt_path"] = args.load_checkpoint

    print("LAUNCHING TRAINING WITH ARGS:")
    pprint(train_args)

    # Train the model
    trainer.fit(**train_args, datamodule=datamodule)
