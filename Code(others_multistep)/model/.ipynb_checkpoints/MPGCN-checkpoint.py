import CONST as _CONST
import PARAM as _PARAM
import torch as _torch
from torch import nn as _nn



I=_torch.eye(_CONST.NUM_ZONEs).to(_PARAM.DEVICE)


def Cosine_Similarity(X):
    l=X.pow(2).sum(-1,keepdim=True).sqrt()
    return (X@X.T)/(l@l.T+1e-8)


def Process_Adjacency(A,kernel_type='symmetric'):
    A=A+I
    if kernel_type=='symmetric':
        D=A.sum(-1).pow(-0.5).diag()
        L=I-D@A@D  # (N,N)
        
        L=(2/_torch.linalg.eigvals(L).abs().max())*L-I
    if kernel_type=='random_walk':
        D=A.sum(-1).pow(-1).diag()
        L=D@A
    L_cheb=_torch.stack([I,L,2*L@L-I])  # (3,N,N)
    return L_cheb  # (3,N,N)



class BDGCN(_nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDGCN, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = _nn.Parameter(_torch.empty(self.input_dim*(self.K**2), self.hidden_dim), requires_grad=True)
        _nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = _nn.Parameter(_torch.empty(self.hidden_dim), requires_grad=True)
            _nn.init.constant_(self.b, val=b_init)
        return

    def forward(self, X:_torch.Tensor, G:_torch.Tensor or tuple):
        feat_set = list()
        if type(G) == _torch.Tensor:         # static graph input: (K, N, N)
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = _torch.einsum('bncl,nm->bmcl', X, G[o, :, :])
                    mode_2_prod = _torch.einsum('bmcl,cd->bmdl', mode_1_prod, G[d, :, :])
                    feat_set.append(mode_2_prod)

        elif type(G) == tuple:              # dynamic graph input: ((batch, K, N, N), (batch, K, N, N))
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = _torch.einsum('bncl,bnm->bmcl', X, G[0][:, o, :, :])
                    mode_2_prod = _torch.einsum('bmcl,bcd->bmdl', mode_1_prod, G[1][:, d, :, :])
                    feat_set.append(mode_2_prod)
        else:
            raise NotImplementedError

        _2D_feat = _torch.cat(feat_set, dim=-1)
        mode_3_prod = _torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            mode_3_prod += self.b
        H = self.activation(mode_3_prod) if self.activation is not None else mode_3_prod
        return H



class MPGCN(_nn.Module):
    def __init__(self,AM_,POI_, M=3, K=2, input_dim=1, lstm_hidden_dim=32, lstm_num_layers=1, gcn_hidden_dim=32, gcn_num_layers=3,
                 num_nodes=_CONST.NUM_ZONEs, user_bias=True, activation=_nn.LeakyReLU(0.1)):
        super(MPGCN, self).__init__()
        
        self.static_graph_am=Process_Adjacency(AM_.to(_PARAM.DEVICE))
        self.static_graph_poi=Process_Adjacency(Cosine_Similarity(POI_.to(_PARAM.DEVICE)))
        
        self.M = M      # input graphs
        self.K = K      # chebyshev order
        self.num_nodes = num_nodes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.gcn_num_layers = gcn_num_layers
        
        # initiate a branch of (LSTM, 2DGCN, FC) for each graph input
        self.branch_models = _nn.ModuleList()
        for m in range(self.M):
            branch = _nn.ModuleDict()
            branch['temporal'] = _nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
            branch['spatial'] = _nn.ModuleList()
            for n in range(gcn_num_layers):
                cur_input_dim = lstm_hidden_dim if n == 0 else gcn_hidden_dim
                branch['spatial'].append(BDGCN(K=K, input_dim=cur_input_dim, hidden_dim=gcn_hidden_dim, use_bias=user_bias, activation=activation))
            branch['fc'] = _nn.Sequential(
                _nn.Linear(in_features=gcn_hidden_dim, out_features=input_dim, bias=True),
                _nn.LeakyReLU(0.1))
            self.branch_models.append(branch)


    def init_hidden_list(self, batch_size:int):     # for LSTM initialization
        hidden_list = list()
        for m in range(self.M):
            weight = next(self.parameters()).data
            hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim),
                      weight.new_zeros(self.lstm_num_layers, batch_size * (self.num_nodes**2), self.lstm_hidden_dim))
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, input):
        '''
        :param x_seq: (batch, seq, O, D, 1)
        :param G_list: static graph (K, N, N); dynamic OD graph tuple ((batch, K, N, N), (batch, K, N, N))
        :return:
        '''
        
        FLOW_=input[-1]
        x_seq=FLOW_.unsqueeze(-1)
        G_list=[self.static_graph_am,
                self.static_graph_poi,
                (_torch.stack([Process_Adjacency(Cosine_Similarity(flow[3:6].mean(-3))) for flow in FLOW_]),
                 _torch.stack([Process_Adjacency(Cosine_Similarity(flow[3:6].mean(-3).T)) for flow in FLOW_]))]
        
        batch_size, seq_len, _, _, i = x_seq.shape
        hidden_list = self.init_hidden_list(batch_size)
        
        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(batch_size*(self.num_nodes**2), seq_len, i)
        branch_out = list()
        for m in range(self.M):
            lstm_out, hidden_list[m] = self.branch_models[m]['temporal'](lstm_in, hidden_list[m])
            gcn_in = lstm_out[:,-1,:].reshape(batch_size, self.num_nodes, self.num_nodes, self.lstm_hidden_dim)
            for n in range(self.gcn_num_layers):
                gcn_in = self.branch_models[m]['spatial'][n](gcn_in, G_list[m])
            fc_out = self.branch_models[m]['fc'](gcn_in)
            branch_out.append(fc_out)
        # ensemble
        ensemble_out = _torch.mean(_torch.stack(branch_out, dim=-1), dim=-1)
        
        return ensemble_out.squeeze(-1)
