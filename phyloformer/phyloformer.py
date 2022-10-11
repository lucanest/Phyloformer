import torch
import torch.nn as nn
from scipy.special import binom
from phyloformer.attentions import KernelAxialMultiAttention

class AttentionNet(nn.Module):
    '''Phyloformer Network'''     
    def __init__(self,dropout=0.0,
    nb_seq=20,seq_len=200,n_blocks=1,device='cpu'):
        super(AttentionNet,self).__init__()
        self.n_blocks=n_blocks
        self.rowAttentions=nn.ModuleList()
        self.columnAttentions=nn.ModuleList()
        self.layernorms=nn.ModuleList()
        self.fNNs=nn.ModuleList()

        layers_1_1=[nn.Conv2d(in_channels=22,out_channels=64, kernel_size=1,stride=1),
        nn.ReLU()]
        self.block_1_1=nn.Sequential(*layers_1_1)
        self.norm=nn.LayerNorm(64)
        self.pwFNN=nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=1, kernel_size=1,stride=1),nn.Dropout(dropout),
        nn.Softplus()])
        for i in range(self.n_blocks):
            self.rowAttentions.append(KernelAxialMultiAttention(64,4,n=seq_len,k=50).to(device))
            self.columnAttentions.append(KernelAxialMultiAttention(64,4,n=int(binom(nb_seq,2)),k=50).to(device))
            self.layernorms.append(nn.LayerNorm(64).to(device))
            self.fNNs.append(nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=256, kernel_size=1,stride=1,device=device),
        nn.GELU(),nn.Conv2d(in_channels=256,out_channels=64, kernel_size=1,stride=1,device=device)]))
            
        self.nb_seq=nb_seq
        self.seq_len=seq_len
        self.nb_pairs=int(binom(nb_seq,2))
        self.device=device

        seq2pair = torch.zeros(self.nb_pairs, self.nb_seq)
        k = 0
        for i in range(self.nb_seq):
            for j in range(i+1, self.nb_seq):
                seq2pair[k, i] = 1
                seq2pair[k, j] = 1
                k = k+1

        self.seq2pair=seq2pair.to(self.device)
        
    def forward(self, x):
        attentionmaps=[]
        out=self.block_1_1(x) #2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        out=torch.matmul(self.seq2pair,out.transpose(-1,-2)) #pair representation

        #from here on the tensor has shape (batch_size,features,nb_pairs,seq_len), all the transpose/permute allow to apply layernorm
        #and attention over the desired dimensions and are then followed by the inverse transposition/permutation of dimensions

        out=self.norm(out.transpose(-1,-3)).transpose(-1,-3) #layernorm

        for i in range(self.n_blocks):
            #AXIAL ATTENTIONS BLOCK
            #----------------------
            #ROW ATTENTION
            att,a=self.rowAttentions[i](out.permute(0,2,3,1))
            out=att.permute(0,3,1,2)+out #row attention+residual connection
            out=self.layernorms[i](out.transpose(-1,-3)).transpose(-1,-3) #layernorm

            #COLUMN ATTENTION
            att,a=self.columnAttentions[i](out.permute(0,3,2,1))

            attentionmaps.append(a)
            out=att.permute(0,3,2,1)+out #column attention+residual connection 
            out=self.layernorms[i](out.transpose(-1,-3)).transpose(-1,-3) #layernorm

            #FEEDFORWARD
            out=self.fNNs[i](out)+out
            if i!=self.n_blocks-1:
                out=self.layernorms[i](out.transpose(-1,-3)).transpose(-1,-3) #layernorm  

        out=self.pwFNN(out)  # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
        out=torch.squeeze(torch.mean(out,dim=-1)) # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
        return out, attentionmaps
