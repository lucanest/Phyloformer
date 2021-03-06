import torch
import argparse
import numpy as np
import os
import skbio
import gc
from ete3 import Tree
from Bio import SeqIO
from scipy.special import binom
from phyloformer import AttentionNet

amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])

def ali_parser(file):
    sequences={}
    for seq_record in SeqIO.parse(file, "fasta"):
        sequences[seq_record.id]=str(seq_record.seq)
    return sequences
def to_array(seq):
    return np.array([(amino_acids==aa).astype(int) for aa in seq])
def configure(net,seqs,device):
    nb_seq=len(seqs)
    seq_len=len(list(seqs.values())[0])
    nb_pairs=int(binom(nb_seq,2))
    net.nb_seq=nb_seq
    net.seq_len=seq_len
    net.nb_pairs=nb_pairs
    seq2pair = torch.zeros(nb_pairs, nb_seq)
    k = 0
    for i in range(net.nb_seq):
        for j in range(i+1, net.nb_seq):
            seq2pair[k, i] = 1
            seq2pair[k, j] = 1
            k = k+1
    net.seq2pair=seq2pair.to(device)

scriptdir=os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('alidir', type=str, help='path to input directory containing the\
.fasta alignments')
parser.add_argument('--o', type=str, default='', help='path to the output directory were the\
.tree tree files will be saved')
parser.add_argument('--m', type=str, default=os.path.join(scriptdir,'models/seqgen_model_state_dict.pt'), help="path to the NN model's state dictionary")
parser.add_argument('--gpu', type=str, default='', help='gpu option')

args = parser.parse_args()
alidir=args.alidir+'/'
outdir=args.o+'/' if len(args.o)>0 else alidir
alignments=[item for item in os.listdir(alidir) if item[-5:]=='fasta']

device = "cuda" if torch.cuda.is_available() and args.gpu!='' else "cpu"
print(f'Working with device={device}')

model = AttentionNet(device=device,n_blocks=6)
model.load_state_dict(torch.load(args.m),strict=True)
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
    configure(model,seqs,device)
    ids=[seq for seq in seqs]
    tensor=tensors[ali][None,:,:]
    y_pred = model(tensor.float())[0]
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
    dm_nn =skbio.DistanceMatrix(dm_nn, ids)
    nn_newick_str =skbio.tree.nj(dm_nn, result_constructor=str)
    t_nn=Tree(nn_newick_str)
    t_nn.write(format=5,outfile=outdir+'predicted_'+ali.split('.')[0]+'.tree')
    gc.collect()
    del tensor, seqs, y_pred, dm_nn
print('Done, predicted trees saved in '+outdir)
