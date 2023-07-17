import argparse
import os

import numpy as np
from dendropy.simulate import treesim
from ete3 import Tree
from tqdm import tqdm


def simulate_trees(numtrees, numleaves, outdir, treeType, bl):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in tqdm(range(numtrees)):
        # Generating the tree topology
        outname = ""
        outname = os.path.join(outdir, str(i) + "_" + str(numleaves) + "_leaves.nwk")
        if treeType == "birth-death":  # using dendropy
            t = treesim.birth_death_tree(
                birth_rate=1.0, death_rate=0.5, num_extant_tips=numleaves
            )
            t.write(path=outname, schema="newick", suppress_rooting=True)
        elif treeType == "uniform":  # using ete3
            t = Tree()
            t.populate(numleaves)
            t.write(format=1, outfile=outname)
        else:
            exit("Error, treetype should be birth-death or uniform")
        t = Tree(outname)

        # Assigning the branch lengths
        for node in t.traverse("postorder"):
            if node.is_root():
                pass
            else:
                if bl == "uniform":
                    node.dist = np.random.uniform(low=0.002, high=1.0, size=None)
                elif bl == "exponential":
                    node.dist = np.random.exponential(0.15, size=None)
                else:
                    exit(
                        "Error, branch length distribution should be uniform or exponential"
                    )
        t.write(format=1, outfile=outname)

    with open(os.path.join(outdir, "stdout.txt"), "a") as fout:
        fout.write(
            f"{numtrees} trees with {numleaves} leaves simulated, topology: {treeType}, branch length distribution: {bl}.\n"
        )


TOPOLOGIES = ["birth-death", "uniform"]
BRLENS = ["exponential", "uniform"]

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        "-n", "--ntrees", type=int, required=False, default=20, help="number of trees"
    )
    parser.add_argument(
        "-l", "--nleaves", type=int, required=False, default=20, help="number of leaves"
    )
    parser.add_argument(
        "-t",
        "--topology",
        type=str,
        required=False,
        default="uniform",
        help=f"tree topology. Choices: {TOPOLOGIES}",
        choices=TOPOLOGIES,
        metavar="TOPO",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=".",
        help="path to the output directory were the .nwk tree files will be saved",
    )
    parser.add_argument(
        "-b",
        "--branchlength",
        type=str,
        required=False,
        default="uniform",
        help=f"branch length distribution. Choices: {BRLENS}",
        choices=BRLENS,
        metavar="BL",
    )
    args = parser.parse_args()

    simulate_trees(
        args.ntrees, args.nleaves, args.output, args.topology, args.branchlength
    )


if __name__ == "__main__":
    main()
