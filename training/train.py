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
MAE= nn.L1Loss()
MRE= lambda x,y: torch.mean(torch.abs(x-y)/x).item()


def kfold(tensors,fold,l):
    if fold==0:
        return tensors[:l],tensors[l:]
    else:
        return(kfold(tensors[l:]+tensors[:l],fold-1,l))

class TensorDataset(Dataset):
    def __init__(self,t,in_dir):
        super(TensorDataset, self).__init__()
        self.t=t
        self.in_dir=in_dir
            
    def __len__(self):
        return len(self.t)

    def __getitem__(self, index):
        return torch.load(self.in_dir+'X'+self.t[index]),torch.load(self.in_dir+'y'+self.t[index])


def train(model,train_loader,test_loader,num_epochs,criterion,optimizer,amp,
        scheduler,device,hyperparameters,fold,
        out_dir,t_losses=[], v_losses=[], v_MAEs=[], v_MREs=[],verbose=True):
        ID=Strip(str(hyperparameters),":{}, '")

        if device=='cuda':
            scaler = GradScaler()
        counter=0
        bestmodel=copy.deepcopy(model)
        model=model.to(device)
        for epoch in range(num_epochs):

            #TRAINING
            model.train()
            train_losses=[]
            for i,batch in enumerate(train_loader):
                x_train, y_train = batch
                x_train, y_train=x_train.to(device), y_train.to(device)
                inputs=x_train.float()

                if device=='cuda' and amp=='true': #Automatic mixed precision
                    with autocast():
                        optimizer.zero_grad()
                        outputs,a_maps=model(inputs)
                        y_train=y_train.type_as(outputs)
                        y_train=torch.squeeze(y_train)
                        train_loss=criterion(outputs, y_train)
                        scaler.scale(train_loss).backward()
                        scaler.unscale_(optimizer)  ###GRADIENT CLIPPING
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, error_if_nonfinite=False)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    optimizer.zero_grad()
                    outputs,a_maps=model(inputs)
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
                    x_test, y_test=x_test.to(device), y_test.to(device)
   
                    model.eval()
                    inputs=x_test.float()
                    if device=='cuda': #Automatic mixed precision
                        with autocast():
                            outputs,a_maps=model(inputs)  
                            y_test=y_test.type_as(outputs)
                            y_test=torch.squeeze(y_test)
                            v_loss=criterion(outputs, y_test).item()
                            v_MAE=MAE(outputs,y_test).item()
                            v_MRE=MRE(y_test,outputs)
                    else:
                        outputs,a_maps=model(inputs)
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
                    bestmodel=copy.deepcopy(model)
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


def main(args):

    in_dir=args.i   
    out_dir=args.o
    config=args.c
    load=args.load

    optimizers={"Adam":torch.optim.Adam,
                "SGD":torch.optim.SGD,
                "AdamW":torch.optim.AdamW}

    # Print title
    with open('title.txt') as f:
        print(f.read())
    start_time=time()
    print('pytorch version', torch.__version__)
    i_time=time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device={device}')
    torch.backends.cudnn.benchmark = True


    e_time=time()
    print('optimizing the network took {:.3f} seconds'.format(e_time-i_time))
    i_time=time()

    # Load the hyperparameters from the config file
    with open(config) as jsonfile:
        hyperparameters=json.load(jsonfile)

    print(f'hyperparameters={hyperparameters}')
    batch_size,epochs,lr, opt,loss,n_blocks,n_folds,n_heads,h_dim,dropout, amp= hyperparameters.values()


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    data=list({item[1:] for item in os.listdir(in_dir) if item[1]=='_'})
    data=data[:]

    e_time=time()
    print('loading the data took {:.3f} seconds'.format(e_time-i_time))
    i_time=time()

    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    random.Random(42).shuffle(data)
    print('shuffling the data took {:.3f} seconds\n'.format(time()-i_time))
        
    l=int(len(data)*0.1)

    models,bestmodels=[],[]

    for fold in range(n_folds):
        print(f'Fold {fold +1}:\n')
        fold_time=time()

        # Create the dataloaders
        test_dataset=TensorDataset(kfold(data,fold,l)[0],in_dir)
        train_loader=DataLoader(dataset=TensorDataset(kfold(data,fold,l)[1],in_dir), batch_size=batch_size, shuffle=True)
        test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


        i_time=time()
        x_test,y_test=next(iter(test_loader))
        e_time=time()-i_time
        print('loading the test dataset took {:.3f} seconds\n'.format(e_time))

        seq_len=x_test.shape[-2]
        nb_seq=int(x_test.shape[-1])
        nb_pairs=int(binom(nb_seq,2))

        model = AttentionNet(seq_len=seq_len,nb_seq=nb_seq,n_blocks=n_blocks,device=device,n_heads=n_heads,h_dim=h_dim,dropout=dropout).float().to(device)

        # Load checkpoint
        if len(load)>0:
            L=torch.load(load,map_location=device)
            model.load_state_dict(L['state_dict'], strict=True)

        models.append(model)
        criterion = nn.MSELoss() if loss=='L2' else nn.L1Loss()

        kwargs={"lr":lr}
        optimizer=optimizers[opt](models[fold].parameters(),**kwargs)
        model_kwargs={}

        if len(load)>0:
            optimizer.load_state_dict(L['optimizer_state_dict'])
            t_losses, v_losses, v_MAEs, v_MREs=itemgetter('train_losses', 'val_losses', 'val_maes', 'val_mres')(L)
            model_kwargs={"t_losses":t_losses,
                            "v_losses":v_losses,
                            "v_MAEs":v_MAEs,
                            "v_MREs":v_MREs}
            del L
        
        #Learning rate schedule
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5,verbose=True)

        #Train the model
        bestmodel,epochs=train(model=models[fold],train_loader=train_loader,test_loader=test_loader,num_epochs=epochs,
                        criterion=criterion,optimizer=optimizer,amp=amp,scheduler=scheduler,device=device,
                        verbose=True,hyperparameters=hyperparameters,
                        out_dir=out_dir, fold=fold,**model_kwargs)
        bestmodels.append(bestmodel)

        print('\nThe fold took {:.3f} seconds\n'.format(time()-fold_time))
   
    print(f'total elapsed time: {time()-start_time} seconds')

    # -------------------------------------------------------------------------- # 
    

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser()    
    parser.add_argument('--i', type=str, help='/path/ to input directory containing the\
    the .pt tensors on which the model will be trained')
    parser.add_argument('--c', type=str, help='/path/ to the configuration json file for the hyperparameters')
    parser.add_argument('--o', type=str, help='/path/ to output directory where the model parameters\
        and the metrics will be saved') 
    parser.add_argument('--load', default="", type=str, help='Load model parameters to train it further')        
    args=parser.parse_args()

    main(args) 