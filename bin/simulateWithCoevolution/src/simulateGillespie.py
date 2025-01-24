import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from ete3 import Tree


# usage:
# python simulateGillespie.py --exchangeabilities ../data/coevolution.txt --eqfreq ../data/coevolution_stationary.txt --tree ../data/200k500gaps100007_20_tips.nwk --seqlen 10 --output ../output
def read_tree_from_file(file):
    with open(file, "r") as f:
        line = ""
        for l in f:
            line += l.strip()
        t = Tree(line, format=1)
    return t


def computeScale(subst_mat, eq_freq):
    diago = np.diag(subst_mat)
    tmp = np.multiply(diago, eq_freq)
    scale = -sum(tmp)
    return scale


def simulateSiteAlongBranch(rate_matrix, starting_state_int, branch_length):
    current_time = 0.0
    current_state_int = starting_state_int
    while current_time < branch_length:
        rate = rate_matrix[current_state_int, current_state_int]
        waiting_time = np.random.exponential(-1 / rate)
        current_time = current_time + waiting_time
        if current_time <= branch_length:
            # print("Substitution at time "+ str(current_time))
            vec = rate_matrix[current_state_int,].flatten()
            vec[current_state_int] = 0.0
            vec = vec / sum(vec)
            current_state_int = np.random.choice(400, 1, p=vec)[0]
            # print("New state : " + str(current_state_int))
    return current_state_int


def sequencesToFasta(sequences, str_states, tree):
    num_seqs = sequences.shape[0]
    seq_len = sequences.shape[1]
    leaves = list()
    sequences_str = list()
    for leaf in tree:
        leaves.append(leaf)
        seq = sequences[leaf.id,]
        seq_str = ">" + leaf.name + "\n"
        for i in seq:
            seq_str += str_states[i]
        seq_str += "\n"
        sequences_str.append(seq_str)
    return sequences_str


def simulate_aln(
    treepath: str, outpath: str, seq_len: int, eq_freq, eq_freq_np, exch_np
):
    starting_sequence = list()
    starting_sequence_int = np.random.choice(400, seq_len, p=eq_freq_np)
    for i in starting_sequence_int:
        starting_sequence.append(eq_freq["state"][i])

    # Computing the complete rate matrix:
    subst_mat = np.multiply(exch_np, eq_freq_np)

    # recompute the diagonal to make sure it is equal to minus the sum of the other terms
    subst_mat = subst_mat - np.diag(subst_mat)
    diago = -subst_mat.sum(1)
    np.fill_diagonal(subst_mat, diago)
    scale = computeScale(subst_mat, eq_freq_np)
    # print("Before rescaling: " + str(scale))

    # rescaling:
    subst_mat = np.multiply(1 / scale, subst_mat)
    scale = computeScale(subst_mat, eq_freq_np)
    # print("After rescaling: " + str(scale))

    # read tree
    tree = read_tree_from_file(treepath)

    id = 0
    for node in tree.traverse("preorder"):
        node.add_features(id=id)
        id = id + 1
    # then simulating for each site of the starting sequence
    numseq = id
    sequences = np.ndarray(shape=(numseq, seq_len), dtype=int)
    sequences[0] = starting_sequence_int
    for site in range(seq_len):
        for node in tree.traverse("preorder"):
            if node.is_root():
                pass
            else:
                sequences[node.id, site] = simulateSiteAlongBranch(
                    subst_mat,
                    sequences.item((node.up.id, site)),
                    branch_length=node.dist,
                )

    seq_str = sequencesToFasta(sequences, eq_freq["state"].tolist(), tree)
    with open(outpath, "w") as fout:
        for s in seq_str:
            fout.write(s)


def main(args):
    exchangeabilities_file = args.exchangeabilities
    eq_freq_file = args.eqfreq
    seq_len = args.seqlen

    exch = pd.read_table(exchangeabilities_file, index_col=0)
    exch_np = exch.to_numpy()
    eq_freq = pd.read_table(eq_freq_file, keep_default_na=False)
    eq_freq_np = eq_freq["prob"].to_numpy().flatten()

    for tree_file in tqdm(os.listdir(args.trees)):
        if not (tree_file.endswith(".nwk") or tree_file.endswith(".newick")):
            continue

        stem = tree_file.removesuffix(".nwk").removesuffix(".newick")

        treepath = os.path.join(args.trees, tree_file)
        outpath = os.path.join(args.output, f"{stem}.fasta")

        simulate_aln(treepath, outpath, seq_len, eq_freq, eq_freq_np, exch_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exchangeabilities",
        type=str,
        help="/path/ to the file containing the exchangeabilities",
    )
    parser.add_argument(
        "--eqfreq", type=str, help="/path/ to the equilibrium frequency file"
    )
    parser.add_argument(
        "--trees", type=str, help="/path/ to the directory containing tree files"
    )
    parser.add_argument("--seqlen", type=int, help="number of sites to simulate")
    parser.add_argument(
        "--output",
        type=str,
        help="/path/ to output directory where the alignment will be saved",
    )
    args = parser.parse_args()

    main(args)
