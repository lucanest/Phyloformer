import argparse
import os
import pickle
import pathlib
import random
import sys
import subprocess
import numpy as np
from pprint import pprint
from Bio import SeqIO
from glob import glob
from tqdm import tqdm

ALPHAS_PATH = os.path.join(os.path.dirname(__file__), "data", "hogenom_alphas.txt")
MAX_ATTEMPTS_DEFAULT = 20


def load_list(listpath):
    with open(listpath, "rb") as file:
        return pickle.load(file)


def sample_scale(scales):
    mean = random.sample(scales, 1)[0]
    scale = np.random.normal(loc=mean, scale=mean / 10)
    return max(scale, 0.05)


def has_duplicates(alnpath):
    seqs = set()
    n_seqs = 0
    for record in SeqIO.parse(alnpath, "fasta"):
        n_seqs += 1
        seqs.add(str(record.seq))
    return n_seqs != len(seqs)


def parse_custom_model_name(modelpath):
    with open(modelpath, "r") as file:
        for line in file:
            if line.startswith("frequency"):
                return line.split()[1].split("_")[0]
    return None


def simulate_alignment(
    treefile,
    substitution,
    gamma,
    custom_model_def,
    custom_model_args,
    outdir,
    length,
    max_attempts,
    threads,
):
    filestem = pathlib.PurePath(treefile).stem
    outpath = os.path.join(outdir, f"{filestem}.fa")

    success = False
    attempt = 1
    while not success:
        if attempt > max_attempts:
            os.remove(outpath)
            return treefile, attempt

        model_args = f"{substitution}"
        if custom_model_args is not None:
            model_args += f"+{custom_model_args}"
        if args.gamma is not None:
            alpha = sample_scale(alphas)
            model_args += f"+{gamma}{{{alpha}}}"

        cmd = [
            "iqtree2",
            "--alisim",
            os.path.join(outdir, filestem),
            "-t",
            treefile,
            "-m",
            model_args,
            "-mwopt",
            "-af",
            "fasta",
            "--seqtype",
            "AA",
            "--length",
            f"{length}",
            "--threads",
            f"{threads}",
            custom_model_def,
        ]

        process = subprocess.Popen(" ".join(cmd), shell=True, stdout=subprocess.PIPE)
        _, error = process.communicate()
        process.wait()

        if error is not None:
            return error, None

        if args.allow_duplicate_sequences:
            break

        if not has_duplicates(outpath):
            success = True

        attempt += 1

    if not args.keep_logfiles:
        logpath = f"{treefile}.log"
        os.remove(logpath)

    return None, None


def wrapper(args):
    return simulate_alignment(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Alignment simulator")
    parser.add_argument(
        "trees",
        type=str,
        help="Path to the directory containing mewick trees",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--length",
        "-l",
        default=500,
        required=False,
        type=int,
        help="Length of the alignment",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        default=None,
        type=str,
        required=False,
        help=(
            "Gamma model for between-site rate heterogeneity "
            "(G[n] for discrete gamma with n categories, GC for continuous gamma)"
        ),
    )
    parser.add_argument(
        "--substitution",
        "-s",
        default="LG",
        type=str,
        required=False,
        help=(
            "Protein substitution model: "
            "classical (LG, WAG, Dayhoff, Blosum62) or mixture (C10, ..., C60)"
        ),
    )
    parser.add_argument(
        "--custom-model",
        "-c",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a custom model definition in the nexus format\n "
            "(e.g. the UDM models in github.com/dschrempf/EDCluster/"
            "Distributions/hogenom/*_lclr_iqtree.nex)"
        ),
    )
    parser.add_argument(
        "--no-summary",
        "-n",
        action="store_true",
        help="If specified suppress the output summarizing which simulation attempts have failed",
    )
    parser.add_argument(
        "--allow-duplicate-sequences",
        "-d",
        action="store_true",
        help="Allow duplicate sequences in the alignments",
    )
    parser.add_argument(
        "--keep-logfiles",
        "-k",
        action="store_true",
        help="Keep IQTree generated log files",
    )
    parser.add_argument(
        "--max-attempts",
        "-m",
        default=MAX_ATTEMPTS_DEFAULT,
        type=int,
        required=False,
        help="Maximum number of attempts to simulate alignment in case of duplicates",
    )
    parser.add_argument(
        "--processes",
        "-p",
        default=1,
        type=int,
        required=False,
        help="Number of threads for alisim to use.",
    )

    args = parser.parse_args()

    pprint(args.__dict__, indent=2)

    alphas = load_list(ALPHAS_PATH)

    os.makedirs(args.outdir)

    # Get custom model params if defined
    custom_model_def = ""
    custom_model_args = ""
    if args.custom_model is not None:
        model_name = parse_custom_model_name(args.custom_model)
        if model_name is None:
            raise ValueError(f"{args.custom_model} is not a valid IQTree model file")
        custom_model_def = f"-mdef {args.custom_model}"
        custom_model_args = model_name

    treefiles = glob(f"{args.trees}/*.n*w*k")  # matches nwk and newick

    failures = []
    for treefile in tqdm(treefiles):
        error, attempt = simulate_alignment(
            treefile,
            args.substitution,
            args.gamma,
            custom_model_def,
            custom_model_args,
            args.outdir,
            args.length,
            args.max_attempts,
            args.processes,
        )
        if error is not None:
            if attempt is None:
                print("Error simulating tree: ", error)
                sys.exit(1)
            failures.append(error)

    if len(failures) > 0 and not args.no_summary:
        print(f"Failed to simulate {len(failures)} alignments without duplicates:")
        for file in failures:
            print(f"  - {file}")
