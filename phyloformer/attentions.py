import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class KernelAxialMultiAttention(nn.Module):
    def __init__(self,h_dim,n_heads,dropout=0.0,eps=1e-6,n=None,k=None):
        super().__init__()

        self.n_heads=n_heads
        self.q_net=nn.Linear(h_dim,h_dim)
        self.k_net=nn.Linear(h_dim,h_dim)
        self.v_net=nn.Linear(h_dim,h_dim)
        self.elu=nn.ELU()
        self.eps=eps


        self.proj_net=nn.Linear(h_dim,h_dim)

        self.att_drop=nn.Dropout(dropout)
        self.proj_drop=nn.Dropout(dropout)

    def forward(self,x):
        Bs=x.shape[0]
        M,T,C=x.shape[-3:]
        N,D=self.n_heads,C//self.n_heads

        q=self.q_net(x).view(Bs,M,T,N,D).transpose(2,3)
        k=self.k_net(x).view(Bs,M,T,N,D).transpose(2,3)
        v=self.v_net(x).view(Bs,M,T,N,D).transpose(2,3)

        q=self.elu(q)+1
        k=self.elu(k)+1

        KtV=k.transpose(-1,-2)@v
        Z=1/(q@k.transpose(-1,-2).sum(dim=-1,keepdim=True)+self.eps)
        Z=Z.expand(Bs,M,N,T,D)
        V=Z@KtV
        V=V.transpose(2,3).contiguous().view(Bs,-1,T,N*D)

        a=None
        out=self.proj_drop(self.proj_net(V))
        return out, a
