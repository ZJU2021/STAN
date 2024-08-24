import CONST as _CONST
import torch as _torch
from torch import nn as _nn



class Conv(_nn.Module):
    def __init__(self,neighbours,dim_in,dim_out):
        super().__init__()
        self.neighbours=neighbours
        self.Linear=_nn.Linear(neighbours.shape[-1]*dim_in,dim_out)
        self.LeakyReLU=_nn.LeakyReLU(0.1)
       
    def forward(self,X):
        if X.dim()==4: return self.LeakyReLU(self.Linear(X[:,:,self.neighbours].flatten(-2)))
        if X.dim()==3: return self.LeakyReLU(self.Linear(X[:,self.neighbours].flatten(-2)))

    
class LSC(_nn.Module):
    def __init__(self,neighbours):
        super().__init__()
        self.Convs_O=_nn.Sequential(Conv(neighbours,_CONST.NUM_ZONEs,16),Conv(neighbours,16,16),Conv(neighbours,16,16))
        self.Convs_D=_nn.Sequential(Conv(neighbours,_CONST.NUM_ZONEs,16),Conv(neighbours,16,16),Conv(neighbours,16,16))
        self.Convs_OD=Conv(neighbours,32,32)
       
    def forward(self,X):
        return self.Convs_OD(_torch.concatenate((self.Convs_O(X),self.Convs_D(X.transpose(-1,-2))),-1))


class TEC(_nn.Module):
    def __init__(self,Cf):
        super().__init__()
        self.Cf=Cf
        self.LSTM=_nn.LSTM(input_size=32*_CONST.NUM_ZONEs,hidden_size=Cf*_CONST.NUM_ZONEs,num_layers=2,batch_first=True)
       
    def forward(self,F_l):
        return self.LSTM(F_l.flatten(-2))[1][0][-1].unflatten(-1,(_CONST.NUM_ZONEs,self.Cf))

    
class GCC(_nn.Module):
    def __init__(self,neighbours,Cf,Cs):
        super().__init__()
        self.Conv=Conv(neighbours,Cf,Cs)
        self.Conv_output=Conv(neighbours,Cf+Cs,_CONST.NUM_ZONEs)
       
    def forward(self,F_lt):
        Fs=self.Conv(F_lt)
        S=(Fs@Fs.transpose(-1,-2)).softmax(-1)
        F_ltg=_torch.concatenate((F_lt,S@Fs),-1)
        return self.Conv_output(F_ltg)


class CSTN(_nn.Module):
    def __init__(self,CD_,num_neighbours=1+8,Cf=16,Cs=16):
        super().__init__()
        neighbours=CD_.sort(-1)[1][:,:num_neighbours]
        self.LSC=LSC(neighbours)
        self.TEC=TEC(Cf)
        self.GCC=GCC(neighbours,Cf,Cs)
        
    def forward(self,input):
        return self.GCC(self.TEC(self.LSC(input[-1])))
