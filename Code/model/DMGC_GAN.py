import CITY
import PATH as _PATH
import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
import os as _os
from torch import nn as _nn
from process import loss as _loss
from time import time as _get_time
from process.loss import Format_Loss as _Format_Loss
from torch.utils.data import DataLoader as _DataLoader
from torchinfo import summary as _summary



I=_torch.eye(_CONST.NUM_ZONEs).to(_PARAM.DEVICE)

def Cosine_Similarity(X):
    l=X.pow(2).sum(-1,keepdim=True).sqrt()
    return (X@X.T)/(l@l.T+1e-8)

def Get_Neighborhood_Relationship(AM_):
    adj=AM_
    return (adj,)

def Get_Mutual_Attraction_Dynamic(FLOW_):
    flow=FLOW_.mean(1)
    adj_O=flow.softmax(-1)
    adj_D=flow.transpose(-1,-2).softmax(-1)
    return (adj_O,adj_D)

def Get_Passenger_Mobility_Association_Mode_Dynamic(FLOW_):
    flow=FLOW_.sum(-1).transpose(-1,-2)
    adj=_torch.stack([Cosine_Similarity(x) for x in flow])
    return (adj,)

def Get_Graphs(AM_,FLOW_):
    As=Get_Neighborhood_Relationship(AM_)+\
       Get_Mutual_Attraction_Dynamic(FLOW_)+\
       Get_Passenger_Mobility_Association_Mode_Dynamic(FLOW_)
    
    A_bars=[I+(1-I)*A for A in As]
    Ds=[_torch.stack([x.diag() for x in A_bar.sum(-1).pow(-0.5)]) for A_bar in A_bars]
    return _torch.concatenate([D@A_bar@D for (A_bar,D) in zip(A_bars,Ds)],1)



class MGC(_nn.Module):
    def __init__(self,dim_MGC):
        super().__init__()
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.Linear=_nn.LazyLinear(dim_MGC)
        
    def forward(self,graphs,H):
        K=graphs.shape[-2]//graphs.shape[-1]
        return H+self.LeakyReLU(self.Linear((graphs.unsqueeze(-3)@H).unflatten(-2,(_CONST.NUM_ZONEs,K)).flatten(-2)))

    
class TMGCN(_nn.Module):
    def __init__(self,dim_MGC,dim_GRU):
        super().__init__()
        self.MGC=MGC(dim_MGC)
        self.GRU=_nn.GRU(dim_GRU,dim_GRU,batch_first=True)
        
    def forward(self,graphs,H):
        return self.GRU(self.MGC(graphs,H).transpose(1,2).flatten(0,1))[0].unflatten(0,(len(H),_CONST.NUM_ZONEs)).transpose(1,2)

    

class Generator(_nn.Module):
    def __init__(self):
        super().__init__()
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.TMGCNs=_nn.ModuleList([TMGCN(dim_MGC=_CONST.NUM_ZONEs,dim_GRU=_CONST.NUM_ZONEs) for i in range(3)])
        self.Linear=_nn.Linear(_CONST.NUM_ZONEs,_CONST.NUM_ZONEs)

    def forward(self,input):
        (AM_,CD_,POI_,FLOW_)=input
        graphs=Get_Graphs(AM_,FLOW_)
        H=FLOW_
        for TMGCN in self.TMGCNs: H=TMGCN(graphs,H)
        return self.LeakyReLU(self.Linear(H[:,-1,:,:]))


class Discriminator(_nn.Module):
    def __init__(self,dim_hidden=256):
        super().__init__()
        self.sequential=_nn.Sequential(_nn.Linear(_CONST.NUM_ZONEs**2,dim_hidden),_nn.LeakyReLU(0.1),
                                       _nn.Linear(dim_hidden,1),_nn.Sigmoid())
        
    def forward(self,OD):
        return self.sequential(OD.flatten(-2))



def Train(model_G,model_D,optimizer_G,optimizer_D,dataloader,loss_fn,metrics,k):
    model_G.train()
    model_D.train()
    n=len(dataloader)
    loss=[]
    t_start=_get_time()
    
    for (batch,data) in enumerate(dataloader):
        (input,target)=data
        fake=model_G([x.to(_PARAM.DEVICE) for x in input])
        real=target.to(_PARAM.DEVICE)
        score=model_D(_torch.concatenate((fake,real)).detach())
        truth=_torch.concatenate((_torch.zeros((len(fake),1)),_torch.ones(len(real),1))).to(_PARAM.DEVICE)
        loss.append(metrics(fake,real))
        
        optimizer_D.zero_grad()
        loss_fn(score,truth).backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        (-model_D(fake).log()).sum().backward()
        optimizer_G.step()
        
        if (1+batch)%k==0:
            t_end=_get_time()
            t=(t_end-t_start)/k
            t_start=t_end
            print(f'{1+batch:>5d}/{n} | {t:4.2f}s/B {t*k:2.0f}s/R {t*n/60:2.0f}m/E [{t*(n-batch-1)/60:2.0f}m] | '+\
                  _Format_Loss([sum(x[-k:])/k for x in zip(*loss)]))

    return [sum(x)/n for x in zip(*loss)]


def Test(model_G,dataloader,loss_fn,metrics):
    model_G.eval()
    n=len(dataloader)
    loss=[]

    with _torch.inference_mode():
        for data in dataloader:
            (input,target)=data
            predict=model_G([x.to(_PARAM.DEVICE) for x in input])
            target=target.to(_PARAM.DEVICE)
            loss.append(metrics(predict.where(predict>0,0),target))
    
    return [sum(x)/n for x in zip(*loss)]


def Loop(model_G,model_D,optimizer_G,optimizer_D,scheduler_G,scheduler_D,dataset,loss_fn,metrics,
         batch_size=16,num_epochs=100,k=100,load_checkpoint=False,change_G_lr=None,change_D_lr=None):
    dataloader={
        'train':_DataLoader(dataset['train'],batch_size=batch_size,shuffle=True,pin_memory=True),
        'validate':_DataLoader(dataset['validate'],batch_size=batch_size,shuffle=True,pin_memory=True),
        'test':_DataLoader(dataset['test'],batch_size=batch_size,shuffle=True,pin_memory=True)
    }
    model_G=model_G.to(_PARAM.DEVICE)
    model_D=model_D.to(_PARAM.DEVICE)
    name='DMGC_GAN'
    path=_PATH.FOLDER_MODEL+f'/{CITY.NAME}/'+name
    if not _os.path.exists(path): _os.mkdir(path)
    start_epoch=1
    
    print(str(_summary(model_G,depth=0)).splitlines()[-4])
    print(str(_summary(model_D,depth=0)).splitlines()[-4])
    if load_checkpoint:
        if _os.listdir(path):
            last_file=path+'/'+sorted(_os.listdir(path))[-1]
            print(f'Load model "{name}" from file "{last_file}".')
            checkpoint=_torch.load(last_file)
            start_epoch+=checkpoint['epoch']
            loss=checkpoint['loss']
            model_G.load_state_dict(checkpoint['model_G_state_dict'])
            model_D.load_state_dict(checkpoint['model_D_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            if change_G_lr: optimizer_G.param_groups[0]["lr"]=change_lr
            if change_D_lr: optimizer_D.param_groups[0]["lr"]=change_lr
            print(checkpoint['parameters'])
            del checkpoint
    print()
    
    for epoch in range(start_epoch,1+num_epochs):
        print(f'Epoch: {epoch:0>3d}/{num_epochs}    Learning Rate: (G){optimizer_G.param_groups[0]["lr"]:.2e} (D){optimizer_D.param_groups[0]["lr"]:.2e}')
        
        loss={'train':Train(model_G,model_D,optimizer_G,optimizer_D,dataloader['train'],loss_fn,metrics,k),
              'validate':Test(model_G,dataloader['validate'],loss_fn,metrics),
              'test':Test(model_G,dataloader['test'],loss_fn,metrics)}
        scheduler_G.step()
        scheduler_D.step()
 
        _torch.save({'epoch':epoch,'loss':loss,
                     'model_G_state_dict':model_G.state_dict(),'optimizer_G_state_dict':optimizer_G.state_dict(),
                     'model_D_state_dict':model_D.state_dict(),'optimizer_D_state_dict':optimizer_D.state_dict(),
                     'parameters':{'LEAKY_RELU_SLOPE':_PARAM.LEAKY_RELU_SLOPE,
                                   'PROMPTER_DEPTH':_PARAM.PROMPTER_DEPTH,
                                   'PROMPTER_WIDTH':_PARAM.PROMPTER_WIDTH,
                                   'DIM_GLOBAL':_PARAM.DIM_GLOBAL,
                                   'DIM_PIPE':_PARAM.DIM_PIPE,
                                   'DIM_EDGE':_PARAM.DIM_EDGE,
                                   'NUM_HEADs':_PARAM.NUM_HEADs,
                                   'NUM_BLOCKs':_PARAM.NUM_BLOCKs,
                                   'DIM_FFN':_PARAM.DIM_FFN,
                                   'DIM_PREDICTOR':_PARAM.DIM_PREDICTOR} if name=='Mine' else 0},
                    path+'/'+f'{name}_{int(_get_time())}_E{epoch:0>3d}__train={loss["train"][0]:5.3f}_validate={loss["validate"][0]:5.3f}_test={loss["test"][0]:5.3f}.pt')

        print(f'   Train Loss:  {_Format_Loss(loss["train"])}\n'+\
              f'Validate Loss:  {_Format_Loss(loss["validate"])}\n'+\
              f'    Test Loss:  {_Format_Loss(loss["test"])}\n')
