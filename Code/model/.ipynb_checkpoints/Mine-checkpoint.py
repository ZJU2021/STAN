import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
from torch import nn as _nn
from torch.nn.functional import leaky_relu as _F

a=0.2


class Prompter(_nn.Module):
    def __init__(self,dim_left,dim_right,fit_E):
        super().__init__()
        self.fit_E=fit_E
        self.S=_nn.Parameter(1e-3*_torch.randn(dim_left,dim_right))
        self.L1=_nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(4*_PARAM.DIM_GLOBAL,_PARAM.DIM_GLOBAL)))
        self.L2=_nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(dim_left,4*_PARAM.DIM_GLOBAL)))
        self.R1=_nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(_PARAM.DIM_GLOBAL,4*_PARAM.DIM_GLOBAL)))
        self.R2=_nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(4*_PARAM.DIM_GLOBAL,dim_right)))

    def forward(self,G):
        P=self.S+_F(self.L2@_F(self.L1@G@self.R1,a)@self.R2,a)
        return P.unsqueeze(-3) if self.fit_E else P


class Prompters(_nn.Module):
    def __init__(self,dim_left,dim_right,direction,fit_E):
        super().__init__()
        self.direction=direction
        n=max(16,min(dim_left,dim_right)//16*16)
        if direction=='R': self.Prompters=_nn.ModuleList([Prompter(dim_left,n,fit_E),Prompter(n,dim_right,fit_E)])
        if direction=='L': self.Prompters=_nn.ModuleList([Prompter(n,dim_right,fit_E),Prompter(dim_left,n,fit_E)])

    def forward(self,G):
        return (self.direction,[Prompter(G) for Prompter in self.Prompters])


def Injector(T,prompters):
    if isinstance(prompters[0],str):
        if prompters[0]=='R': return _F(_F(T@prompters[1][0],a)@prompters[1][1],a)
        if prompters[0]=='L': return _F(prompters[1][1]@_F(prompters[1][0]@T,a),a)
        if prompters[0]=='R_long':
            for i in range(len(prompters[1])): T=_F(T@prompters[1][i],a)
            return T
        if prompters[0]=='L_long':
            for i in range(len(prompters[1])): T=_F(prompters[1][i]@T,a)
            return T
    else:
        if prompters[0][0]=='L' and prompters[1][0]=='R':
            return _F(prompters[0][1][1]@_F(prompters[0][1][0]@T@prompters[1][1][0],a)@prompters[1][1][1],a)
        if prompters[1][0]=='L' and prompters[0][0]=='R':
            return _F(prompters[1][1][1]@_F(prompters[1][1][0]@T@prompters[0][1][0],a)@prompters[0][1][1],a)



class Projector(_nn.Module):
    def __init__(self,dim_global_in,dim_edge_in,init=0.5/_PARAM.DIM_GLOBAL):
        super().__init__()
        self.init=init
        
        self.GLOBAL_to_G=_nn.ParameterList([_nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(dim_global_in,4*_PARAM.DIM_GLOBAL))),
                                            _nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(4*_PARAM.DIM_GLOBAL,8*_PARAM.DIM_GLOBAL))),
                                            _nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(8*_PARAM.DIM_GLOBAL,4*_PARAM.DIM_GLOBAL))),
                                            _nn.Parameter(_nn.init.kaiming_normal_(_torch.empty(4*_PARAM.DIM_GLOBAL,_PARAM.DIM_GLOBAL)))])
        
        self.pre_project_EDGE=Prompters(dim_edge_in,_PARAM.DIM_EDGE//2,'R',True)
        self.pre_project_FLOW=Prompters(_PARAM.NUM_HISTORY,_PARAM.DIM_EDGE//2,'R',True)

    def forward(self,GLOBAL,EDGE,FLOW):
        G=_torch.stack([x.diag() for x in Injector(self.init*GLOBAL,['R_long',self.GLOBAL_to_G])])
        E=_torch.concatenate((Injector(EDGE,self.pre_project_EDGE(G)),Injector(FLOW.permute(0,2,3,1),self.pre_project_FLOW(G))),-1)
        return (G,E)



class Update_O(_nn.Module):
    def __init__(self):
        super().__init__()
        self.W_L=Prompters(_CONST.NUM_ZONEs,_CONST.NUM_ZONEs,'L',True)
        self.W_R=Prompters(_PARAM.DIM_EDGE,_PARAM.DIM_EDGE,'R',True)
        
    def forward(self,G,E):
        return Injector(E.transpose(1,2),(self.W_L(G),self.W_R(G))).transpose(1,2)

class Update_D(_nn.Module):
    def __init__(self):
        super().__init__()
        self.W_L=Prompters(_CONST.NUM_ZONEs,_CONST.NUM_ZONEs,'L',True)
        self.W_R=Prompters(_PARAM.DIM_EDGE,_PARAM.DIM_EDGE,'R',True)
       
    def forward(self,G,E):
        return Injector(E,(self.W_L(G),self.W_R(G)))

    
class Update(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Update_O=Update_O()
        self.Update_D=Update_D()
        self.W=Prompters(2*_PARAM.DIM_EDGE,_PARAM.DIM_EDGE,'R',True)
       
    def forward(self,G,E):
        return Injector(_torch.concatenate((self.Update_O(G,E),self.Update_D(G,E)),-1),self.W(G))


class Attention_O(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Aggregation=Prompters(_PARAM.DIM_EDGE,1,'R',True)

    def forward(self,G,E):
        A=Injector(E.transpose(1,2),self.Aggregation(G)).squeeze(-1).softmax(-1).unsqueeze(-3)
        return (A@E.transpose(1,2)).transpose(1,2)

class Attention_D(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Aggregation=Prompters(_PARAM.DIM_EDGE,1,'R',True)

    def forward(self,G,E):
        A=Injector(E,self.Aggregation(G)).squeeze(-1).softmax(-1).unsqueeze(-3)
        return A@E

    
class Attention(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Attention_O=Attention_O()
        self.Attention_D=Attention_D()
        self.W=Prompters(2*_PARAM.DIM_EDGE,_PARAM.DIM_EDGE,'R',True)
        
    def forward(self,G,E):
        return Injector(_torch.concatenate((self.Attention_O(G,E),self.Attention_D(G,E)),-1),self.W(G))


class MHA(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Attentions=_nn.ModuleList([Attention() for i in range(_PARAM.NUM_HEADs)])
        self.Weights=Prompter(_PARAM.NUM_HEADs,1,True)
        
    def forward(self,G,E):
        return (_torch.stack(([Attention(G,E) for Attention in self.Attentions]),-1)@self.Weights(G).softmax(-2).unsqueeze(-3)).squeeze(-1)
    


class FFN(_nn.Module):
    def __init__(self):
        super().__init__()
        self.W=Prompters(_PARAM.DIM_EDGE,_PARAM.DIM_EDGE,'R',True)
        
    def forward(self,G,E):
        return Injector(E,self.W(G))



class Transformer_Block(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Update=Update()
        self.MHA=MHA()
        self.FFN=FFN()
        
    def forward(self,G,E):
        E=E+self.Update(G,E)
        E=E+self.MHA(G,E)
        E=E+self.FFN(G,E)
        return E



class Transformer(_nn.Module):
    def __init__(self):
        super().__init__()
        self.Transformer_Blocks=_nn.ModuleList([Transformer_Block() for i in range(_PARAM.NUM_BLOCKs)])

    def forward(self,G,E):
        for Transformer_Block in self.Transformer_Blocks: E=Transformer_Block(G,E)
        return E



class Mine(_nn.Module):
    def __init__(self,dim_global_in,dim_edge_in):
        super().__init__()
        self.Projector=Projector(dim_global_in,dim_edge_in)
        self.Transformer=Transformer()
        self.W=Prompters(_PARAM.DIM_EDGE,1,'R',True)
        
    def forward(self,input):
        (G,E)=self.Projector(*input)
        return Injector(self.Transformer(G,E),self.W(G)).squeeze(-1)
