import os
import ete3
import numpy as np
from ete3 import Tree

scriptdir=os.path.dirname(os.path.realpath(__file__))
trues=os.path.join(scriptdir,'trees')
preds=os.path.join(scriptdir,'alignments')
true_trees=[item for item in os.listdir(trues) if item[-4:]=='tree']
predicted_trees=[item for item in os.listdir(preds) if item[-4:]=='tree']
RFs=[]
for tree in true_trees:
    t1=Tree(os.path.join(trues,tree))
    t2=Tree(os.path.join(preds,'predicted_'+tree))
    RFs.append(t1.compare(t2,unrooted=True)['norm_rf'])
print(f'Mean normalized Robinson-Foulds distance between true and predicted trees: {np.mean(RFs)}')