import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
from torch import nn as _nn



num_nodes=_CONST.NUM_ZONEs**2
I=_torch.eye(num_nodes)


def Cosine_Similarity(X):
    l=X.pow(2).sum(-1,keepdim=True).sqrt()
    return (X@X.T)/(l@l.T+1e-8)


def Get_Neighborhood_Relationship(AM_):
    print('Get_Neighborhood_Relationship')
    n=len(AM_)
    l=[(i,j) for i in range(n) for j in range(n)]
    pairs=AM_.flatten()
    adj_O=pairs[[n*x[0]+y[0] for y in l for x in l]].reshape((len(l),len(l)))
    adj_D=pairs[[n*x[1]+y[1] for y in l for x in l]].reshape((len(l),len(l)))

    return (adj_O,adj_D)

def Get_Functional_Similarity(POI_):
    print('Get_Functional_Similarity')
    n=len(POI_)
    l=[(i,j) for i in range(n) for j in range(n)]
    pairs=Cosine_Similarity(POI_).flatten()
    adj_O=pairs[[n*x[0]+y[0] for y in l for x in l]].reshape((len(l),len(l)))
    adj_D=pairs[[n*x[1]+y[1] for y in l for x in l]].reshape((len(l),len(l)))
    
    return (adj_O,adj_D)

def Get_Centroid_Distance(CD_):
    print('Get_Centroid_Distance')
    n=len(CD_)
    l=[(i,j) for i in range(n) for j in range(n)]
    pairs=CD_.flatten()
    adj_O=pairs[[n*x[0]+y[0] for y in l for x in l]].reshape((len(l),len(l)))
    adj_D=pairs[[n*x[1]+y[1] for y in l for x in l]].reshape((len(l),len(l)))
    
    return (adj_O,adj_D)

def Get_Mobility_Pattern_Correlation(FLOW_):
    print('Get_Mobility_Pattern_Correlation')
    n=FLOW_.shape[-1]
    l=[(i,j) for i in range(n) for j in range(n)]
    pairs1=Cosine_Similarity(FLOW_.permute(1,2,0).flatten(1)).flatten()
    pairs2=Cosine_Similarity(FLOW_.permute(2,1,0).flatten(1)).flatten()
    adj_O1=pairs1[[n*x[0]+y[0] for y in l for x in l]].reshape((len(l),len(l)))
    adj_D1=pairs1[[n*x[1]+y[1] for y in l for x in l]].reshape((len(l),len(l)))
    adj_O2=pairs2[[n*x[0]+y[0] for y in l for x in l]].reshape((len(l),len(l)))
    adj_D2=pairs2[[n*x[1]+y[1] for y in l for x in l]].reshape((len(l),len(l)))
    
    return (adj_O1,adj_D1,adj_O2,adj_D2)


def Get_Graphs(AM_,CD_,POI_,FLOW_):
    As=Get_Neighborhood_Relationship(AM_)+\
       Get_Functional_Similarity(POI_)+\
       Get_Centroid_Distance(CD_)+\
       Get_Mobility_Pattern_Correlation(FLOW_)
    A_bars=[I+(1-I)*A for A in As]
    Ds=[A_bar.sum(-1).pow(-0.5).diag() for A_bar in A_bars]
    print('Get_Graphs OK')
    return _torch.concatenate([D@A_bar@D for (A_bar,D) in zip(A_bars,Ds)])
    

    
class MGC(_nn.Module):
    def __init__(self,graphs,dim_in,dim_out):
        super().__init__()
        self.graphs=graphs
        self.K=graphs.shape[0]//graphs.shape[1]
        self.Linear=_nn.Linear(self.K*dim_in,dim_out)
        
    def forward(self,H):
        return self.Linear((self.graphs@H).unflatten(-2,(num_nodes,self.K)).flatten(-2))


class RMGC(_nn.Module):
    def __init__(self,graphs,dim_in,dim_out):
        super().__init__()
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        self.conv_block=_nn.Sequential(MGC(graphs,dim_in,dim_out//2),_nn.LeakyReLU(0.1),
                                       MGC(graphs,dim_out//2,dim_out//2),_nn.LeakyReLU(0.1),
                                       MGC(graphs,dim_out//2,dim_out))
        self.res_part=_nn.Sequential(MGC(graphs,dim_out,dim_out))
        self.identity_block=_nn.Sequential(MGC(graphs,dim_out,dim_out//2),_nn.LeakyReLU(0.1),
                                           MGC(graphs,dim_out//2,dim_out//2),_nn.LeakyReLU(0.1),
                                           MGC(graphs,dim_out//2,dim_out))
        
    def forward(self,H):
        conv_block_out=self.conv_block(H)
        res_out=self.res_part(conv_block_out)
        conv_part_out=conv_block_out+res_out
        conv_part_out=self.LeakyReLU(conv_part_out)
        identity_out=self.identity_block(conv_part_out)
        out=self.LeakyReLU(conv_part_out+identity_out)
        return out


class ST_ED_RMGC(_nn.Module):
    def __init__(self,AM_,CD_,POI_,FLOW_,
                 input_dim=sum(_PARAM.W_D_P),output_dim=1,
                 enc_feature=32,dec_feature=8,fc_enc_feature=4,L1=128,
                 rnn_feat=256,num_rnn_layers=2,L2=128):
        super().__init__()
        graphs=Get_Graphs(AM_,CD_,POI_,FLOW_).to(_PARAM.DEVICE)
        self.LeakyReLU=_nn.LeakyReLU(0.1)
        
        self.input_graph_encoder=RMGC(graphs,input_dim,enc_feature)
        self.input_graph_encoder_flatten=_nn.Linear(num_nodes*enc_feature,L1)
        
        self.input_time_fc=_nn.Linear(num_nodes,num_nodes//2)
        self.input_time_encoder=_nn.LSTM(num_nodes//2,rnn_feat,num_layers=num_rnn_layers)
        self.input_time_encoder_flatten=_nn.Linear(rnn_feat,L2)
        self.input_time_encoder_output=_nn.Linear(rnn_feat,num_nodes)

        self.rebuild_fc=_nn.Linear(L1+L2,num_nodes*fc_enc_feature)
        
        self.decoder=RMGC(graphs,fc_enc_feature,dec_feature)
        self.decoder_output_layer=MGC(graphs,dec_feature,output_dim)
        self.decoder_one_output_layer=MGC(graphs,fc_enc_feature,output_dim)

        self.non_decoder_output_layer=_nn.Linear(L1+L2,num_nodes)
        
    def forward(self,input):
        flow=input[-1].flatten(-2).transpose(-1,-2)
        graph_encoder_output=self.input_graph_encoder(flow)
        graph_encoder_output_flatten=self.input_graph_encoder_flatten(graph_encoder_output.reshape((len(flow),-1)))
        input_time_fc=self.input_time_fc(flow.transpose(1,2).transpose(0,1))
        lstm_output,(hidden,cell)=self.input_time_encoder(input_time_fc)
        graph_time_output_flatten=self.LeakyReLU(self.input_time_encoder_flatten(hidden[-1]))
        fusion=_torch.cat([graph_encoder_output_flatten,graph_time_output_flatten],dim=1)
        output=self.non_decoder_output_layer(fusion)
        
        return output.unflatten(-1,(_CONST.NUM_ZONEs,_CONST.NUM_ZONEs))
