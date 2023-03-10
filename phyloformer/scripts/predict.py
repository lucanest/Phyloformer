import argparse
import os
import gc

import skbio
import torch
import numpy as np

from collections import OrderedDict

from Bio import SeqIO
from ete3 import Tree
from scipy.special import binom

from phyloformer.phyloformer import AttentionNet

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

def phyloformer_predict(alidir, model_path, device, outdir, save_dm):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(f'Working with device={device}')
    alignments = [item for item in os.listdir(alidir) if item[-5:]=='fasta']
    model=AttentionNet(device=device,n_blocks=6)

    loaded=torch.load(model_path,map_location=device)
    state_dict = loaded["state_dict"] if loaded.get("state_dict") is not None else loaded

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
        id_seq=ali_parser(os.path.join(alidir, ali))
        arrays=[to_array(seq) for seq in id_seq.values()]
        for array in arrays:
                X.append(((torch.from_numpy(array)).t().view(22,1,-1)))
        X=torch.cat(X,dim=1).transpose(-1,-2)
        tensors[ali]=X.to(device)
        
    for ali in tensors:
        print(f'processing alignment {ali}...')
        counter=0
        seqs=ali_parser(os.path.join(alidir, ali))
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
                    pred=y_pred[counter].item()
                    pred=float('%.6f'%(pred))
                    nn_dist[(i,j)]=pred
                    nn_dist[(j,i)]=nn_dist[(i,j)]
                    counter+=1
        dm_nn=[[nn_dist[(i,j)] for j in range(len(seqs))] for i in range(len(seqs))]
        dm_nn=skbio.DistanceMatrix(dm_nn, ids)

        if save_dm:
            save_dm(dm_nn, os.path.join(outdir, ali.split('.')[0]+'.pf.dm'))

        nn_newick_str=skbio.tree.nj(dm_nn, result_constructor=str)
        t_nn=Tree(nn_newick_str)
        t_nn.write(format=5,outfile=os.path.join(outdir, ali.split('.')[0]+'.pf.nwk'))
        gc.collect()
        del tensor,seqs,y_pred,dm_nn

    print(f'Done, predicted trees saved in {outdir}')


def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('alidir', type=str, help='path to input directory containing the\
    .fasta alignments')
    parser.add_argument('-o', '--output', type=str, required=False, help='path to the output directory were the\
    .tree tree files will be saved (default: alidir)')
    parser.add_argument('-m', '--model', type=str, required=False, default=os.path.join(scriptdir,"..", "pretrained_models", "seqgen_model_state_dict.pt"), help="path to the NN model's state dictionary (default: pretrained model)")
    parser.add_argument('-g', '--gpu', required=False, action="store_true", help='use the GPU for inference (default: false)')
    parser.add_argument('-d', '--dm',  required=False, action="store_true", help='save predicted distance matrix (default: false)')
    args = parser.parse_args()

    outdir = args.output if args.output is not None else args.alidir
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

    print(args)

    phyloformer_predict(args.alidir, args.model, device, outdir, args.dm)


if __name__ == "__main__":
    main()
