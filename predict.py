import torch
import argparse
import numpy as np
import os
import skbio
import gc
from ete3 import Tree
from Bio import SeqIO
from scipy.special import binom
from phyloformer.phyloformer import AttentionNet
from collections import OrderedDict

amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])

def save_dm(dm,d_file):
    with open(d_file,'w+') as fp:
        fp.write(str(len(dm.ids))+'\n')
        ids=list(dm.ids)
        for i,dist in enumerate(dm.data):
            dist=[str(item) for item in dist]
            dist=' '.join(dist)
            fp.write(str(ids[i])+'     '+dist+'\n')

def ali_parser(file):
    sequences={}
    for seq_record in SeqIO.parse(file, "fasta"):
        sequences[seq_record.id]=str(seq_record.seq)
    return sequences
def to_array(seq):
    return np.array([(amino_acids==aa).astype(int) for aa in seq])
def configure(model,seqs,device):
    nb_seq=len(seqs)
    seq_len=len(list(seqs.values())[0])
    nb_pairs=int(binom(nb_seq,2))
    model.nb_seq=nb_seq
    model.seq_len=seq_len
    model.nb_pairs=nb_pairs
    seq2pair=torch.zeros(nb_pairs, nb_seq)
    k=0
    for i in range(model.nb_seq):
        for j in range(i+1, model.nb_seq):
            seq2pair[k, i] = 1
            seq2pair[k, j] = 1
            k = k+1
    model.seq2pair=seq2pair.to(device)
    del seq2pair

scriptdir=os.path.dirname(os.path.realpath(__file__))
parser=argparse.ArgumentParser()
parser.add_argument('alidir', type=str, help='path to input directory containing the\
.fasta alignments')
parser.add_argument('--o', type=str, default='', help='path to the output directory were the\
.tree tree files will be saved')
parser.add_argument('--m', type=str, default=os.path.join(scriptdir,'models/seqgen_model_state_dict.pt'), help="path to the NN model's state dictionary")
parser.add_argument('--gpu', type=str, default='', help='gpu option')
parser.add_argument('--dm', type=str, default='', help='option to save predicted distance matrix')

args=parser.parse_args()
alidir=args.alidir+'/'
outdir=args.o+'/' if len(args.o)>0 else alidir
alignments=[item for item in os.listdir(alidir) if item[-5:]=='fasta']

device="cuda" if torch.cuda.is_available() and args.gpu!='' else "cpu"
print(f'Working with device={device}')

model=AttentionNet(device=device,n_blocks=6)
state_dict=torch.load(args.m,map_location=device)
new_state_dict=OrderedDict()
for k,v in state_dict.items():
    name=k.replace("module.", "")  #remove "module." for models trained on multiple gpus
    new_state_dict[name]=v
model.load_state_dict(new_state_dict,strict=True)
model.to(device)
model.eval()
tensors={}

for ali in alignments:
    X=[]
    id_seq=ali_parser(alidir+ali)
    arrays=[to_array(seq) for seq in id_seq.values()]
    for array in arrays:
            X.append(((torch.from_numpy(array)).t().view(22,1,-1)))
    X=torch.cat(X,dim=1).transpose(-1,-2)
    tensors[ali]=X.to(device)
    
for ali in tensors:
    print(f'processing alignment {ali}...')
    counter=0
    seqs=ali_parser(alidir+ali)
    if len(seqs)!=model.nb_seq:
        configure(model,seqs,device)
    ids=[seq for seq in seqs]
    tensor=tensors[ali][None,:,:]
    with torch.no_grad():
        y_pred=model(tensor.float())[0]
    y_pred=y_pred.view(model.nb_pairs)
    nn_dist={}
    for i,leaf1 in enumerate(seqs):
        for j,leaf2 in enumerate(seqs):
            if i==j:
                nn_dist[(i,j)]=0
            if i<j:
                nn_dist[(i,j)]=y_pred[counter].item()
                nn_dist[(j,i)]=nn_dist[(i,j)]
                counter+=1
    dm_nn=[[nn_dist[(i,j)] for j in range(len(seqs))] for i in range(len(seqs))]
    dm_nn=skbio.DistanceMatrix(dm_nn, ids)
    if args.dm=='true':
        save_dm(dm_nn,outdir+ali.split('.')[0]+'.pf.dm')
    nn_newick_str=skbio.tree.nj(dm_nn, result_constructor=str)
    t_nn=Tree(nn_newick_str)
    t_nn.write(format=5,outfile=outdir+ali.split('.')[0]+'.pf.nwk')
    gc.collect()
    del tensor,seqs,y_pred,dm_nn
print('Done, predicted trees saved in '+outdir)
