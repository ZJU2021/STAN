import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
from torch import nn as _nn



I=_torch.eye(_CONST.NUM_ZONEs).to(_PARAM.DEVICE)


class Pre_Weighted_Aggregator(_nn.Module):
    def __init__(self,AM_,dim_in,dim_out):
        super().__init__()
        am=AM_.to(_PARAM.DEVICE)
        am=am*(1-I)
        self.w_r=am/(am.sum(-1,keepdim=True)+1e-8)
        
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.W_r=_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_in,dim_out)))
        self.W_s=_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_in,dim_out)))

    def forward(self,X,FLOW_):
        flow=FLOW_+FLOW_.transpose(-1,-2)
        flow=flow*(1-I)
        w_s=flow/(flow.sum(-1,keepdim=True)+1e-8)
        
        w_r=self.w_r
        return _torch.concatenate((self.LeakyReLU((X+w_r@X)@self.W_r),
                                   self.LeakyReLU((X+w_s@X)@self.W_s)),-1)



class GEML(_nn.Module):
    def __init__(self,AM_,POI_,dim_in=_CONST.NUM_ZONEs+_CONST.NUM_CATEGORYs,dim_PWA=256,dim_LSTM=256):
        super().__init__()
        self.POI_=POI_.to(_PARAM.DEVICE)
        self.Pre_Weighted_Aggregator=Pre_Weighted_Aggregator(AM_,dim_in,dim_PWA)
        self.LSTM=_nn.LSTM(input_size=2*dim_PWA,hidden_size=dim_LSTM,num_layers=3,batch_first=True)
        self.W=_nn.Parameter(_nn.init.orthogonal_(_torch.empty(dim_LSTM,dim_LSTM)))

    def forward(self,input):
        FLOW_=input[-1]
        X0=_torch.concatenate((FLOW_,_torch.stack([_torch.stack([self.POI_ for i in range(sum(_PARAM.W_D_P))]) for j in range(len(FLOW_))])),-1)  # (B,L,N,dim_in)
        # (B,L,N,dim_PWA) --> (B,N,L,dim_PWA) --> (B*N,L,dim_PWA) --> (B*N,dim_LSTM) --> (B,N,dim_LSTM)
        X=self.LSTM(self.Pre_Weighted_Aggregator(X0,FLOW_).transpose(1,2).flatten(0,1))[1][0][-1].unflatten(0,(len(FLOW_),_CONST.NUM_ZONEs))
        return X@self.W@X.transpose(-1,-2)
