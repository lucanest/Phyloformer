import torch
import torch.nn as nn
import os, sys, random
import argparse
import numpy as np
import json
import copy
import math
import sys
sys.path.insert(0,"..")
from phyloformer.phyloformer import AttentionNet
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from scipy.special import binom
from time import time
from operator import itemgetter

Strip=lambda string,chars: Strip(string.replace(chars[0],""),chars[1:]) if len(chars)>0 else string

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

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # Print title
    with open('title.txt') as f:
        print(f.read())

    start_time=time()
    print('pytorch version', torch.__version__)

    parser = argparse.ArgumentParser()    
    parser.add_argument('--i', type=str, help='/path/ to input directory containing the\
    the .pt tensors on which the model will be trained')
    parser.add_argument('--c', type=str, help='/path/ to the configuration json file for the hyperparameters')
    parser.add_argument('--o', type=str, help='/path/ to output directory where the model parameters\
        and the metrics will be saved') 
    parser.add_argument('--load', default="", type=str, help='Load model parameters to train it further')        
    args=parser.parse_args()

    in_dir=args.i   
    out_dir=args.o
    config=args.c
    load=args.load

    i_time=time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device={device}')
    torch.backends.cudnn.benchmark = True

    print('optimizing the network took {:.3f} seconds'.format(time()-i_time))
    i_time=time()

    # Load the hyperparameters from the config file
    with open(config) as jsonfile:
        hyperparameters=json.load(jsonfile)

    print(f'hyperparameters={hyperparameters}')
    batch_size,epochs,lr, opt,loss,n_blocks,n_folds= hyperparameters.values()

    ID=Strip(str(hyperparameters),":{}, '")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    data=list({item[1:] for item in os.listdir(in_dir) if item[1]=='_'})

    print('loading the data took {:.3f} seconds'.format(time()-i_time))
    i_time=time()

    random.Random(42).shuffle(data)
    print('shuffling the data took {:.3f} seconds\n'.format(time()-i_time))
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    l=int(len(data)*0.1)
    
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
                inputs=x_train.float()

                if device=='cuda': #Automatic mixed precision
                    with autocast():
                        optimizer.zero_grad()
                        outputs,a_maps=net(inputs)
                        y_train=y_train.type_as(outputs)
                        y_train=torch.squeeze(y_train)
                        train_loss=criterion(outputs, y_train)
                        scaler.scale(train_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    optimizer.zero_grad()
                    outputs,a_maps=net(inputs)
                    y_train=y_train.type_as(outputs)
                    y_train=torch.squeeze(y_train)
                    train_loss=criterion(outputs, y_train)
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

            t_losses.append(np.mean(train_losses))

            #TESTING
            with torch.no_grad():
                val_MAEs=[]
                val_losses=[]
                val_MREs=[]
                for j,batch in enumerate(test_loader):
                    x_test, y_test = batch
   
                    net.eval()
                    inputs=x_test.float()
                    if device=='cuda': #Automatic mixed precision
                        with autocast():
                            outputs,a_maps=net(inputs)  
                            y_test=y_test.type_as(outputs)
                            y_test=torch.squeeze(y_test)
                            v_loss=criterion(outputs, y_test).item()
                            v_MAE=MAE(outputs,y_test).item()
                            v_MRE=MRE(y_test,outputs)
                    else:
                        outputs,a_maps=net(inputs)
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
                v_MAEs.append(val_MAE)
                v_MREs.append(val_MRE)
                v_losses.append(val_loss)
                scheduler.step(val_loss)
        
                #Retain the model with the best performances so far
                if epoch==0:
                    bestloss=val_loss
                if epoch>0 and val_loss<bestloss:
                    counter=0
                    bestloss=val_loss
                    print(f'best val loss {val_loss} at epoch {epoch+1}')
                    bestmodel=copy.deepcopy(net)
                    torch.save({'state_dict':bestmodel.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'hyperparameters':hyperparameters,
                    'val_losses':v_losses,
                    'train_losses':t_losses,
                    'val_maes':v_MAEs,
                    'val_mres':v_MREs},
                    out_dir+'/Model'+ID+'fold'+str(fold+1)+'best.pt')

                else:
                    counter+=1    
                if counter>8 or math.isnan(val_loss):
                    #print(counter,val_loss)
                    return bestmodel, epoch+1

            if (epoch%5==0 or epoch==num_epochs-1) and verbose:
                print(f'\nepoch={epoch+1}, train loss={t_losses[-1]:.6f}, '
                f'val loss={val_loss:.6f}, val MAE={val_MAE:.6f}, val MRE={val_MRE:.6f}.\n')

        return bestmodel, num_epochs

    nets,bestnets=[],[]

    for fold in range(n_folds):
        print(f'Fold {fold +1}:\n')
        fold_time=time()

        # Create the dataloaders
        test_dataset=TensorDataset(kfold(data,fold,l)[0],in_dir,device)
        train_loader=DataLoader(dataset=TensorDataset(kfold(data,fold,l)[1],in_dir,device), batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        i_time=time()
        x_test,y_test=next(iter(test_loader))
        print('loading the test dataset took {:.3f} seconds\n'.format(time()-i_time))

        seq_len=x_test.shape[-2]
        nb_seq=int(x_test.shape[-1])
        nb_pairs=int(binom(nb_seq,2))

        net = AttentionNet(seq_len=seq_len,nb_seq=nb_seq,n_blocks=n_blocks,device=device).float().to(device)
        
        if len(load)>0:
            L=torch.load(load)
            net.load_state_dict(L['state_dict'],strict=True)

        nets.append(net)
        criterion = nn.MSELoss() if loss=='L2' else nn.L1Loss()
        MAE= nn.L1Loss()
        MRE= lambda x,y: torch.mean(torch.abs(x-y)/x).item()
    
        if opt == 'Adam':
            optimizer = torch.optim.Adam(nets[fold].parameters(),lr=lr)
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(nets[fold].parameters(),lr=lr, weight_decay=0,momentum=0.9)
        else:
            raise ValueError('Please specify either Adam or SGD as optimizer')
        
        if len(load)>0:
            optimizer.load_state_dict(L['optimizer_state_dict'])

        if len(load)>0:
            t_losses, v_losses, v_MAEs, v_MREs=itemgetter('train_losses', 'val_losses', 'val_maes', 'val_mres')(L)

        else: 
            t_losses, v_losses, v_MAEs, v_MREs = ([] for i in range(4))

        #Learning rate schedule
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5,verbose=True)

        #Train the model
        bestnet,epochs=train(nets[fold],epochs,verbose=True)
        bestnets.append(bestnet)

        print('\nThe fold took {:.3f} seconds\n'.format(time()-fold_time))
   

    #Save the models
    for i,net in enumerate(nets):
        torch.save({'state_dict':net.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'hyperparameters':hyperparameters,
        'val_losses':v_losses,
        'train_losses':t_losses,
        'val_maes':v_MAEs,
        'val_mres':v_MREs},
        out_dir+'/Model'+ID+'fold'+str(i+1)+'.pt')

    print(f'total elapsed time: {time()-start_time} seconds')

if __name__ == '__main__':
    main() 
