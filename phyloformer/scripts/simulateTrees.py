import argparse
import os
import numpy as np
from dendropy.simulate import treesim
from ete3 import Tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nleaves', type=int, default=20, help='number of leaves')
    parser.add_argument('--ntrees', type=int, help='number of trees')
    parser.add_argument('--type', type=str, default='uniform', help='topology: birth-death or uniform')
    parser.add_argument('--o', type=str, default='', help='path to the output directory were the\
    .nwk tree files will be saved')
    parser.add_argument('--bl',type=str, default='uniform',help='branch length distribution: uniform or exponential')

    args = parser.parse_args()
    numleaves = args.nleaves
    numtrees = args.ntrees
    treeType = args.type
    outdir = args.o
    bl=args.bl

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(numtrees):
        #Generating the tree topology
        outname = ""
        outname = os.path.join(outdir, str(i) + "_" + str(numleaves) + "_leaves.nwk")
        if treeType == "birth-death": # using dendropy
            t = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=numleaves)
            t.write(path=outname, schema="newick", suppress_rooting=True)
        elif treeType == "uniform": # using ete3
            t = Tree()
            t.populate(numleaves)
            t.write(format=1, outfile=outname)
        else:
            exit("Error, treetype should be birth-death or uniform")
        t=Tree(outname)

        #Assigning the branch lengths
        for node in t.traverse("postorder"):
            if node.is_root():
                pass
            else:
                if bl=='uniform':
                    node.dist = np.random.uniform(low=0.002, high=1.0, size=None)
                elif bl=='exponential':
                    node.dist = np.random.exponential(0.15, size=None)
                else:
                    exit("Error, branch length distribution should be uniform or exponential")
        t.write(format=1, outfile=outname)


    with open(os.path.join(outdir,  "stdout.txt"), 'a') as fout:
            fout.write(f"{numtrees} trees with {numleaves} leaves simulated, topology: {treeType}, branch length distribution: {bl}.\n")


if __name__ == "__main__":
    main()
