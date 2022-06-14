from numpy.core.fromnumeric import shape
import torch
from torch._C import device
import torch.nn as nn
import os, random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from torch.utils.data import Dataset, DataLoader
from scipy.special import binom
from time import time
import copy
import math
from torch.cuda.amp import autocast, GradScaler
#from torchsummary import summary

Strip=lambda string,chars: Strip(string.replace(chars[0],""),chars[1:]) if len(chars)>0 else string
normalize= lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
pairs=lambda n: [[i,j] for i in range(n) for j in range(i+1,n)]

amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])

def indexes(k,n):
    a=pairs(n)
    i,j=a[k]
    return [a.index(element) for element in a if i in element or j in element]

def savemaps(a,out_dir,label=''):
    '''Saves attention maps as figures'''
    for j,item in enumerate(a):
        im1, im2, im3, im4=[item.cpu().numpy()[i] for i in range(item.shape[0])]
        fig = plt.figure(figsize=(40., 40.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                axes_pad=0.8,  # pad between axes in inch.
                )
        for ax, im in zip(grid, [im1, im2, im3, im4]):
            ax.imshow(im,cmap='gray')
        fig.savefig(out_dir+'/AmapsBlock'+str(j+1)+'__'+label+'.png')
        plt.close('all')

def plot(models,out_dir,epochs,ID,start_epoch=0,label=''):
    '''Plotting'''
    print()
    train_losses,val_losses,val_MAEs=[],[],[]
    for i,model in enumerate(models):
        train_losses.append([t for t in model.train_losses][start_epoch:])
        val_losses.append([t for t in model.val_losses][start_epoch:])
        val_MAEs.append([t for t in model.val_MAEs][start_epoch:])

    mean_train_losses=[sum(x)/len(x) for x in zip(*train_losses)]
    mean_val_losses=[sum(x)/len(x) for x in zip(*val_losses)]
    mean_val_MAEs=[sum(x)/len(x) for x in zip(*val_MAEs)]

    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))
    ax1.plot(range(start_epoch,epochs),train_losses[0],label='training (first fold) '+label,linewidth=2)
    ax1.plot(range(start_epoch,epochs),val_losses[0],color='r', label='validation (first fold) '+label,linewidth=2)
    if len(models)>1:
        ax1.plot(range(start_epoch,epochs),mean_val_losses,color='k',label='validation (mean across folds) '+label,linewidth=1.5)
        for i,model in enumerate(models):
            if i==0:
                ax1.plot(range(start_epoch,epochs),val_losses[i],color='gray',label='validation (other folds)',linewidth=1.5,alpha=0.2)
            else:
                ax1.plot(range(start_epoch,epochs),val_losses[i],color='gray',linewidth=1.5,alpha=0.2)

    ax1.title.set_text('Loss (Mean squared error)')
    ax1.legend(loc='upper left')
    annotate_min(list(range(start_epoch,epochs)), mean_val_losses, ax=ax1)

    ax2.plot(range(start_epoch,epochs),val_MAEs[0],color='navy', label='validation (first fold)',linewidth=2)
    ax2.plot(range(start_epoch,epochs),mean_val_MAEs,color='k',label='validation (mean across folds)',linewidth=1.5)
    for i,model in enumerate(models):
        if i==0:
            ax2.plot(range(start_epoch,epochs),val_MAEs[i],color='gray',label='validation (other folds)',linewidth=1.5,alpha=0.2)
        else:
            ax2.plot(range(start_epoch,epochs),val_MAEs[i],color='gray',linewidth=1.5,alpha=0.2)    
            
    ax2.title.set_text('Mean absolute error')
    ax2.legend(loc='upper left')
    annotate_min(list(range(start_epoch,epochs)), mean_val_MAEs, ax=ax2)
    plt.savefig(out_dir+'attention'+ID+'from_epoch'+str(start_epoch)+'.png')
    plt.close('all')
    #plt.show()

def annotate_min(epochs,loss, ax):
        epochs=np.array(epochs)
        loss=np.array(loss)
        xmin = epochs[np.argmin(loss)]
        ymin = loss.min()
        text= "epoch={:.0f}, value={:.3f}".format(xmin, ymin)
        bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.8)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)

def fulltensortoseqs(x):
    sliced=[x[i:i+22,:] for i in range(0,x.shape[0],22)]
    sliced=[np.transpose(np.array(item)) for item in sliced]
    aa_indexes=[[np.where(item[i]==1)[0][0] for i in range(x.shape[1])] for item in sliced]
    seqs=[''.join(list(amino_acids[item])) for item in aa_indexes]
    return seqs

def hamming(s,t):
    '''Computes Hamming distance between two sequences'''
    dist=0
    if len(s)!=len(t):
        print('different lenghts!')
        return
    for i, char in enumerate(s):
        if s[i]!=t[i]:
            dist+=1       
    return dist  

def kfold(tensors,fold,l):
    if fold==0:
        return tensors[:l],tensors[l:]
    else:
        return(kfold(tensors[l:]+tensors[:l],fold-1,l))

class TensorDataset(Dataset):

    def __init__(self,t,in_dir,device):
        super(TensorDataset, self).__init__()
        self.t=t
        self.in_dir=in_dir
        self.device=device
            
    def __len__(self):
        return len(self.t)

    def __getitem__(self, index):
        return torch.load(self.in_dir+'X'+self.t[index]).to(self.device),torch.load(self.in_dir+'y'+self.t[index]).to(self.device)

class AxialMultiAttention(nn.Module):
    def __init__(self,h_dim,n_heads,device,dropout=0.0,tied=False,linear=False,random=True,n=None,k=None):
        super().__init__()

        self.tied=tied
        self.linear=linear
        self.n_heads=n_heads
        self.q_net=nn.Linear(h_dim,h_dim)
        self.k_net=nn.Linear(h_dim,h_dim)
        self.v_net=nn.Linear(h_dim,h_dim)

        if self.linear:
            self.n=n    #WHEN USING LINEAR WE HAVE TO PASS BOTH THE INPUT DIMENSION (NUMBER OF VECTORS)
            self.k=k    #AND THE REDUCED DIMENSION
            if random:
                self.delta=1/2**n
                self.R=(torch.randn((n,k))*1/k).to(device)
                self.linK=self.R*self.delta
                self.linV=self.R/(torch.e**(-self.delta))
            
            else:
                self.linK=nn.Linear(n,k)
                self.linV=nn.Linear(n,k)
        
        self.proj_net=nn.Linear(h_dim,h_dim)

        self.att_drop=nn.Dropout(dropout)
        self.proj_drop=nn.Dropout(dropout)

    def forward(self,x):
        Bs=x.shape[0]
        M,T,C=x.shape[-3:]
        N,D=self.n_heads,C//self.n_heads

        q=self.q_net(x).view(Bs,M,T,N,D).transpose(2,3)

        if self.linear:
            if random:
                k=(self.k_net(x).transpose(-1,-2)@self.linK).transpose(-1,-2).view(Bs,M,self.k,N,D).transpose(2,3)
                v=(self.v_net(x).transpose(-1,-2)@self.linV).transpose(-1,-2).view(Bs,M,self.k,N,D).transpose(2,3)
            
            else:
                k=self.linK(self.k_net(x).transpose(-1,-2)).transpose(-1,-2).view(Bs,M,self.k,N,D).transpose(2,3)
                v=self.linV(self.v_net(x).transpose(-1,-2)).transpose(-1,-2).view(Bs,M,self.k,N,D).transpose(2,3)
        else:
            k=self.k_net(x).view(Bs,M,T,N,D).transpose(2,3)
            v=self.v_net(x).view(Bs,M,T,N,D).transpose(2,3)

        if self.tied:
            q=torch.sum(q,dim=1,keepdim=True)
            k=torch.sum(k,dim=1,keepdim=True)
            weights=q@k.transpose(-2,-1)/math.sqrt(D*M)
        else:
            weights=q@k.transpose(3,4)/math.sqrt(D)
        weights=F.softmax(weights,dim=-1)
        attention=self.att_drop(weights@v)
        attention=attention.transpose(2,3).contiguous().view(Bs,-1,T,N*D)
        #idxs=torch.argsort(torch.mean(weights.view(Bs,-1,T),dim=1))[:,int(T/4):].int().tolist()
        out=self.proj_drop(self.proj_net(attention))
        a=weights.detach()[0].mean(0)
        return out, a#, idxs
        #return torch.stack([out[i,:,idxs[i],:] for i in range(len(idxs))]), a

class SparseColumnAttention(nn.Module):
    def __init__(self,h_dim,n_heads,device,dropout=0.0,n=10,k=None):
        super().__init__()

        self.n_heads=n_heads
        self.q_net=nn.Linear(h_dim,h_dim)
        self.k_net=nn.Linear(h_dim,h_dim)
        self.v_net=nn.Linear(h_dim,h_dim)
        self.proj_net=nn.Linear(h_dim,h_dim)

        self.att_drop=nn.Dropout(dropout)
        self.proj_drop=nn.Dropout(dropout)
        self.pair_idxs=np.array([indexes(j,n) for j in range(int(binom(n,2)))])
        self.n=n

    def forward(self,x):
        Bs=x.shape[0]
        M,T,C=x.shape[-3:]
        N,D=self.n_heads,C//self.n_heads

        q=self.q_net(x).view(Bs,M,T,N,D).transpose(2,3)
        k=self.k_net(x).view(Bs,M,T,N,D).transpose(2,3)
        v=self.v_net(x).view(Bs,M,T,N,D).transpose(2,3)

        q=q.view(Bs,M,N,T,1,D)
        self.pair_idxs=np.array([indexes(j,10) for j in range(int(binom(10,2)))])
        self.n=10
        print(k.shape,v.shape,self.pair_idxs.shape)
        k=k.contiguous().view(-1,T,D)[:,self.pair_idxs].contiguous().view(Bs,M,N,T,2*self.n-3,D)
        v=v.contiguous().view(-1,T,D)[:,self.pair_idxs].contiguous().view(Bs,M,N,T,2*self.n-3,D)

        #print(k.shape)
        weights=q@k.transpose(-1,-2)/math.sqrt(D)
        #print(weights.shape)
        weights=F.softmax(weights,dim=-1)
        attention=self.att_drop(weights@v)
        attention=attention.transpose(2,3).contiguous().view(Bs,-1,T,N*D)
        #print(attention.shape)

        out=self.proj_drop(self.proj_net(attention))
        a=None
        return out, a

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

class CrossAxialMultiAttention(nn.Module):
    def __init__(self,h_dim,n_heads,dropout=0.0,tied=False):
        super().__init__()

        self.tied=tied
        self.n_heads=n_heads
        self.q_net=nn.Linear(h_dim,h_dim)
        self.k_net=nn.Linear(h_dim,h_dim)
        self.v_net=nn.Linear(h_dim,h_dim)

        self.proj_net=nn.Linear(h_dim,h_dim)

        self.att_drop=nn.Dropout(dropout)
        self.proj_drop=nn.Dropout(dropout)

    def forward(self,x,y):
        Bs=x.shape[0]
        M,T,C=x.shape[-3:]
        N,D=self.n_heads,C//self.n_heads

        q=self.q_net(x).view(Bs,M,T,N,D).transpose(2,3)
        k=self.k_net(y).view(Bs,M,-1,N,D).transpose(2,3) 
        v=self.v_net(y).view(Bs,M,-1,N,D).transpose(2,3)

        if self.tied:
            q=torch.sum(q,dim=1,keepdim=True)
            k=torch.sum(k,dim=1,keepdim=True)
            weights=q@k.transpose(-2,-1)/math.sqrt(D*M)
        else:
            weights=q@k.transpose(3,4)/math.sqrt(D)
        normalized_weights=F.softmax(weights,dim=-1)
        attention=self.att_drop(normalized_weights@v)
        attention=attention.transpose(2,3).contiguous().view(Bs,-1,T,N*D)

        out=self.proj_drop(self.proj_net(attention))
        a=normalized_weights.detach()[0].mean(0)
        return out, a

class AttentionNet(nn.Module):
    '''Phyloformer Network'''     
    def __init__(self,dropout=0.0,
    batch_norm=False,nb_seq=20,seq_len=200,n_blocks=1,device=device,crossattention=False,linear=False):
        super(AttentionNet,self).__init__()
        self.crossattention=crossattention
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
            if crossattention:
                self.columnAttentions.append(CrossAxialMultiAttention(64,4).to(device))
            else:
                self.columnAttentions.append(KernelAxialMultiAttention(64,4,n=int(binom(nb_seq,2)),k=50).to(device))
            self.layernorms.append(nn.LayerNorm(64).to(device))
            self.fNNs.append(nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=256, kernel_size=1,stride=1,device=device),
        nn.GELU(),nn.Conv2d(in_channels=256,out_channels=64, kernel_size=1,stride=1,device=device)]))
            
        self.train_losses=[]
        self.val_losses=[]
        self.mean_val_losses=[]
        self.val_MAEs=[]
        self.val_MREs=[]
        self.val_predictions=[]
        self.train_predictions=[]
        self.nb_seq=nb_seq
        self.seq_len=seq_len
        self.nb_pairs=int(binom(nb_seq,2))
        self.train_history=[]
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
        #print(x.shape)
        attentionmaps=[]
        out=self.block_1_1(x) #2d convolution that gives us the features in the third dimension
        #print(out.shape)
        out=torch.matmul(self.seq2pair,out.transpose(-1,-2)) #pair representation
        #print(out.shape)

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
            if self.crossattention:
                att,a=self.columnAttentions[i](out.permute(0,3,2,1),seqs.permute(0,3,2,1))
            else:
                att,a=self.columnAttentions[i](out.permute(0,3,2,1))

            attentionmaps.append(a)
            out=att.permute(0,3,2,1)+out #column attention+residual connection 
            out=self.layernorms[i](out.transpose(-1,-3)).transpose(-1,-3) #layernorm

            #FEEDFORWARD
            #print('feedforward')
            out=self.fNNs[i](out)+out
            if i!=self.n_blocks-1:
                out=self.layernorms[i](out.transpose(-1,-3)).transpose(-1,-3) #layernorm  

        out=self.pwFNN(out)  # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
        #stds=1-0.97*normalize(torch.squeeze(torch.std(out,dim=-1)).detach().cpu().numpy())
        out=torch.squeeze(torch.mean(out,dim=-1)) # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)   
        #out=out[:,0,:,0] #SPECIAL COLUMN FOR PREDICTION
        return out, attentionmaps#, stds

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    start_time=time()
    print('pytorch version', torch.__version__)

    parser = argparse.ArgumentParser()    
    parser.add_argument('indir', type=str, help='/path/ to input directory containing the\
    .dat alignments or the .pt tensors on which the model will be trained')
    parser.add_argument('config', type=str, help='/path/ to the configuration json file for the hyperparameters')
    parser.add_argument('outdir', type=str, help='/path/ to output directory where the model\
        and the plots of his performances will be saved') 
    parser.add_argument('--label', default="", type=str, help='Label for the saved models/plots') 
    parser.add_argument('--load', default="", type=str, help='Load model to train it further')        
    args = parser.parse_args()

    in_dir = args.indir   
    out_dir = args.outdir
    Label=args.label
    config=args.config
    load=args.load

    i_time=time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.backends.cudnn.benchmark = True

    print('optimizing the network took {:.3f} seconds'.format(time()-i_time))
    i_time=time()

    # Load the hyperparameters from the config file
    with open(config) as jsonfile:
        hyperparameters=json.load(jsonfile)

    print(f'hyperparameters={hyperparameters}')
    batch_size, epochs, lr, opt, nb_folds, ds,n_blocks,linear= hyperparameters.values()
    linear=linear=='true'

    ID=Strip(str(hyperparameters),":{}, '")+Label
    os.mkdir(out_dir+'/'+ID)
    out_dir=out_dir+'/'+ID+'/'


    data=list({item[1:] for item in os.listdir(in_dir) if item[1]=='_'})
    #data=data[:ds]

    print('loading the data took {:.3f} seconds'.format(time()-i_time))
    i_time=time()

    random.Random(42).shuffle(data)
    print('shuffling the data took {:.3f} seconds\n'.format(time()-i_time))
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    l=int(len(data)*0.1)
    plotmaps=False
    
    # -------------------------------------------------------------------------- # 

    def train(net,num_epochs,verbose=True):
        if device=='cuda':
            scaler = GradScaler()
        counter=0
        bestmodel=copy.deepcopy(net)
        net=net.to(device)
        for epoch in range(num_epochs):

            #TRAINING
            net.train()
            train_losses=[]
            for i,batch in enumerate(train_loader):
                x_train, y_train = batch
                if len(x_train.shape)==4:
                        y_train=torch.squeeze(y_train[:,0,:,:])      #For quartets as we have them saved with all the permutations
                        x_train=x_train[:,0,:,:]
                        if epoch==0 and i==0:
                            print('Working with quartets')    

                inputs=x_train.float()

                if device=='cuda': #MIXED PRECISION
                    with autocast():
                        optimizer.zero_grad()
                        outputs,a_maps = net(inputs)
                        y_train=y_train.type_as(outputs)
                        y_train=torch.squeeze(y_train)
                        train_loss = criterion(outputs, y_train)
                        scaler.scale(train_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    optimizer.zero_grad()
                    outputs,a_maps = net(inputs)
                    y_train=y_train.type_as(outputs)
                    y_train=torch.squeeze(y_train)
                    train_loss = criterion(outputs, y_train)
                    train_loss.backward()
                    optimizer.step()
                train_losses.append(train_loss.item())

                #Clear memory
                del batch
                del x_train
                del y_train
                del inputs
                del outputs
                del a_maps

            net.train_losses.append(np.mean(train_losses))

            #TESTING
            with torch.no_grad():
                val_MAEs=[]
                val_losses=[]
                val_MREs=[]
                for j,batch in enumerate(test_loader):
                    x_test, y_test = batch
                    if len(x_test.shape)==4:
                        y_test=torch.squeeze(y_test[:,0,:,:])      #For quartets as we have them saved with all the permutations
                        x_test=x_test[:,0,:,:]     

                    net.eval()
                    inputs=x_test.float()
                    if device=='cuda': #MIXED PRECISION
                        with autocast():
                            outputs,a_maps = net(inputs)  
                            if (epoch%5==0 or epoch==num_epochs-1) and j==0 and plotmaps:
                                savemaps(a_maps,out_dir,'epoch'+str(epoch))
                            #if device=="cuda":
                            #    print(torch.cuda.memory_summary())
                            y_test=y_test.type_as(outputs)
                            y_test=torch.squeeze(y_test)
                            v_loss=criterion(outputs, y_test).item()
                            v_MAE=MAE(outputs,y_test).item()
                            v_MRE=MRE(y_test,outputs)
                    else:
                        outputs,a_maps = net(inputs)  
                        if (epoch%5==0 or epoch==num_epochs-1) and j==0 and plotmaps:
                            savemaps(a_maps,out_dir,'epoch'+str(epoch))
                        #if device=="cuda":
                        #    print(torch.cuda.memory_summary())
                        y_test=y_test.type_as(outputs)
                        y_test=torch.squeeze(y_test)
                        v_loss=criterion(outputs, y_test).item()
                        v_MAE=MAE(outputs,y_test).item()
                        v_MRE=MRE(y_test,outputs)

                    val_losses.append(v_loss)
                    val_MAEs.append(v_MAE)
                    val_MREs.append(v_MRE)

                    #Clear memory
                    del batch
                    del x_test
                    del y_test
                    del inputs
                    del outputs
                    del a_maps

                val_MAE=np.mean(val_MAEs)
                val_MRE=np.mean(val_MREs)
                val_loss=np.mean(val_losses)
                net.val_MAEs.append(val_MAE)
                net.val_MREs.append(val_MRE)
                net.val_losses.append(val_loss)
                scheduler.step(val_loss)
        
                #Retain the model with the best performances so far
                if epoch==0:
                    bestloss=val_loss
                if epoch>0 and val_loss<bestloss:
                    counter=0
                    bestloss=val_loss
                    print(f'best val loss {val_loss} at epoch {epoch+1}')
                    bestmodel=copy.deepcopy(net)
                    torch.save(bestmodel,out_dir+'Attention'+ID+'fold'+Label+'best.pt') 

                else:
                    counter+=1    
                if counter>8 or math.isnan(val_loss):
                    print(counter,val_loss)
                    return bestmodel, epoch+1

            plot(nets,epochs=epoch+1,out_dir=out_dir,ID=ID)

            if (epoch%5==0 or epoch==num_epochs-1) and verbose:
                print(f'\n epoch={epoch+1}, train loss={net.train_losses[-1]:.6f}, '
                f'val loss={val_loss:.6f}, val MAE={val_MAE:.6f}, val MRE={val_MRE:.6f}.')

        return bestmodel, num_epochs


    nets,bestnets=[],[]

    for fold in range(nb_folds):
        print(f'Fold {fold +1}:\n')
        fold_time=time()

        # Create the dataloaders

        test_dataset=TensorDataset(kfold(data,fold,l)[0],in_dir,device)
        train_loader=DataLoader(dataset=TensorDataset(kfold(data,fold,l)[1],in_dir,device), batch_size=batch_size, shuffle=True,num_workers=0)
        test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        i_time=time()
        x_test,y_test=next(iter(test_loader))
        print('loading the test dataset took {:.3f} seconds'.format(time()-i_time))

        seq_len=x_test[0].shape[-1]
        nb_seq=int(x_test[0].shape[-2]/22)
        nb_pairs=int(binom(nb_seq,2))
        
        if load!="":
            net=torch.load(load,map_location=torch.device(device))
        else:
            net=AttentionNet(seq_len=seq_len,nb_seq=nb_seq,n_blocks=n_blocks,device=device,linear=linear).float().to(device)
            #print(summary(net,x_test.shape))
        
        nets.append(net)
        criterion = nn.MSELoss()
        MAE= nn.L1Loss()
        MRE= lambda x,y: torch.mean(torch.abs(x-y)/x).item()

        if opt == 'Adam':
            optimizer = torch.optim.Adam(nets[fold].parameters(),lr=lr)
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(nets[fold].parameters(),lr=lr, weight_decay=0,momentum=0.9)
        else:
            raise ValueError('Please specify either Adam or SGD as optimizer')
        
        #Learning rate schedule
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5,verbose=True)

        #Train the model
        bestnet,epochs=train(nets[fold],epochs,verbose=True) #FIX FOR MULTIPLE FOLDS
        bestnets.append(bestnet)
        #Save the hyperparameters with which the model has been trained so far
        bestnets[fold].train_history.append(hyperparameters)
        nets[fold].train_history.append(hyperparameters)

        print('the fold took {:.3f} seconds\n'.format(time()-fold_time))

    #Save the models
    for i,net in enumerate(nets):
        torch.save(net,out_dir+'Attention'+ID+'fold'+str(i+1)+Label+'.pt')
    for i,net in enumerate(bestnets):   
        torch.save(net,out_dir+'Attention'+ID+'fold'+str(i+1)+Label+'.pt') 
    plot(nets,epochs=epochs,out_dir=out_dir,ID=ID)

    print(f'total elapsed time: {time()-start_time} seconds')

if __name__ == '__main__':
    main() 
