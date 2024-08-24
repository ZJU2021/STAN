import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
from torch import nn as _nn



class LSTM(_nn.Module):
    def __init__(self,):
        super().__init__()
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.LSTM=_nn.LSTM(input_size=1,hidden_size=128,num_layers=1,batch_first=True,proj_size=1)
        
    def forward(self,input):
        h=self.LSTM(input[-1].permute(0,2,3,1).flatten(0,2).unsqueeze(-1))[1][0][-1]
        return self.LeakyReLU(h).squeeze(-1).unflatten(0,(len(input[-1]),_CONST.NUM_ZONEs,_CONST.NUM_ZONEs))

    
    
class Spatial_LSTM(_nn.Module):
    def __init__(self,):
        super().__init__()
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.LSTM=_nn.LSTM(input_size=_CONST.NUM_ZONEs**2,hidden_size=256,num_layers=2,batch_first=True)
        self.Project=_nn.Linear(256,_CONST.NUM_ZONEs**2)
        
    def forward(self,input):
        output=self.Project(self.LSTM(input[-1].flatten(-2))[1][0][-1])
        return self.LeakyReLU(output).unflatten(-1,(_CONST.NUM_ZONEs,_CONST.NUM_ZONEs))

    

class GCN(_nn.Module):
    def __init__(self,AM_,dim_in=2*sum(_PARAM.W_D_P)*_CONST.NUM_ZONEs,dim_hidden=256,dim_out=_CONST.NUM_ZONEs,num_layers=3):
        super().__init__()
        A=AM_+_torch.eye(_CONST.NUM_ZONEs)
        D=A.sum(1).pow(-0.5).diag()
        self.DAD=(D@A@D).to(_PARAM.DEVICE)
        
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.Ws=_nn.ParameterList([_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_in,dim_hidden)))]+\
                                  [_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_hidden,dim_hidden))) for i in range(num_layers-2)]+\
                                  [_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_hidden,dim_hidden)))])
        self.W=_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_hidden,dim_hidden)))
        
    def forward(self,input):
        H=_torch.concatenate((input[-1].permute(0,2,3,1).flatten(-2),input[-1].permute(0,3,2,1).flatten(-2)),-1)
        for W in self.Ws: H=self.LeakyReLU(self.DAD@H@W)
        return self.LeakyReLU(H@self.W@H.transpose(-1,-2))
