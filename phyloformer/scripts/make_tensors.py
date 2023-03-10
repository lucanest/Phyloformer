import argparse
import os

import numpy as np
import torch

from itertools import permutations

from Bio import SeqIO
from ete3 import Tree
from scipy.special import binom

amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])

def ali_parser(file):
    sequences={}
    for seq_record in SeqIO.parse(file, "fasta"):
        sequences[seq_record.id]=str(seq_record.seq)
    return sequences

def tree_dist(t,normalization=False):
    distances={}
    for i,leaf1 in enumerate(t):
        for j,leaf2 in enumerate(t):
            if i<j:
                distances[(leaf1.name,leaf2.name)]=leaf1.get_distance(leaf2)
    if normalization:
        diam=max(distances.values())
        for dist in distances:
            distances[dist]/=diam
    return(distances)

def to_array(seq):
    return np.array([(amino_acids==aa).astype(int) for aa in seq])

def make_tensors(tree_path, ali_path, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    trees=[item[:-4] for item in os.listdir(tree_path) if item[-4:]=='.nwk']

    for tree in trees:

        t=Tree(os.path.join(tree_path,tree+'.nwk'))
        alignment=os.path.join(ali_path,tree+'.fasta')

        print(f'Processing {tree}')

        distances=tree_dist(t)

        X,y=[],[]
        id_seq=ali_parser(alignment)
        arrays=[to_array(seq) for seq in id_seq.values()]
        for array in arrays:
            X.append(((torch.from_numpy(array)).t().view(22,1,-1)))
        X=torch.cat(X,dim=1).transpose(-1,-2)
        for id1,id2 in distances:
            y.append(distances[(id1,id2)])

        y=torch.tensor(y)
        torch.save(X,os.path.join(out_dir,'X_'+tree+'.pt'))
        torch.save(y,os.path.join(out_dir,'y_'+tree+'.pt'))

    print('Done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--treedir', required=True, type=str,help='path to input directory containing the .nwk tree files')
    parser.add_argument('-a', '--alidir', required=True, type=str, help='path to input directory containing corresponding .fasta alignments')
    parser.add_argument('-o', '--output', required=False, default=".", type=str, help='path to output directory (default: current directory)')
    args = parser.parse_args()

    make_tensors(args.treedir, args.alidir, args.output)

if __name__ == '__main__':
    main()
