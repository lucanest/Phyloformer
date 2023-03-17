import argparse
import os

import torch
from tqdm import tqdm

from phyloformer.data import load_alignment, load_tree


def make_tensors(tree_dir: str, aln_dir: str, out_dir: str):

    for tree_file in (pbar := tqdm(os.listdir(tree_dir))):
        identifier = tree_file.rstrip(".nwk")
        pbar.set_description(f"Processing {identifier}")
        tree_tensor, _ = load_tree(os.path.join(tree_dir, tree_file))
        aln_tensor, _ = load_alignment(os.path.join(aln_dir, f"{identifier}.fasta"))

        torch.save(
            {"X": aln_tensor, "y": tree_tensor},
            os.path.join(out_dir, f"{identifier}.tensor_pair"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a tensor training set from trees and MSAs"
    )
    parser.add_argument(
        "-t",
        "--treedir",
        required=True,
        type=str,
        help="path to input directory containing the .nwk tree files",
    )
    parser.add_argument(
        "-a",
        "--alidir",
        required=True,
        type=str,
        help="path to input directory containing corresponding .fasta alignments",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=".",
        type=str,
        help="path to output directory (default: current directory)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    make_tensors(args.treedir, args.alidir, args.output)


if __name__ == "__main__":
    main()
