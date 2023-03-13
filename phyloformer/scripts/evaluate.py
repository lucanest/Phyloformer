import argparse
import os

import numpy as np
from ete3 import Tree


def evaluate(trues, preds):
    true_trees=[item for item in os.listdir(trues) if item[-3:]=='nwk']
    RFs=[]
    for tree in true_trees:
        t1=Tree(os.path.join(trues, tree))
        t2=Tree(os.path.join(preds, tree.split('.nwk')[0]+'.pf.nwk'))
        RFs.append(t1.compare(t2,unrooted=True)['norm_rf'])
    print(f'Mean normalized Robinson-Foulds distance between true and predicted trees: {np.mean(RFs):.3f}')

def main():
    parser = argparse.ArgumentParser(description="Compute the RF distance between predicted trees and true trees.")
    parser.add_argument("-t", "--true", required=True, type=str, help="path to directory containing true trees in .nwk format")
    parser.add_argument("-p", "--predictions", required=True, type=str, help="path to directory containing predicted trees in .nwk format")
    args = parser.parse_args()

    evaluate(args.true, args.predictions)


if __name__ == "__main__":
    main()
