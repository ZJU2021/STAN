import warnings as _warnings
import CITY
import CONST as _CONST
import torch as _torch
from utils.converter import N_to_XY
from utils import loader as _loader
from sklearn.metrics.pairwise import haversine_distances as _haversine_distances



def Temporal_Encoding():
    if CITY.NAME=='NYC': slots=_torch.arange(_CONST.NUM_SLOTs_year)
    if CITY.NAME=='NB': slots=_torch.arange(_CONST.NUM_SLOTs)
    p_week=2*_torch.pi*(((slots+_CONST.T_start.day_of_week*_CONST.NUM_SLOTs_day)%_CONST.NUM_SLOTs_week)/_CONST.NUM_SLOTs_week)
    p_day=2*_torch.pi*((slots%_CONST.NUM_SLOTs_day)/_CONST.NUM_SLOTs_day)
    return _torch.stack([_torch.cos(p_week),_torch.sin(p_week),_torch.cos(p_day),_torch.sin(p_day)]).transpose(0,1).float()


def Adjacency_Matrix():
    return _torch.tensor(_loader.Load_Adjacency(_CONST.SCOPE)).unsqueeze(-1).float()


def Centroid_Distance():
    if CITY.NAME=='NYC':
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            ZONE=_loader.Load_Zone(_CONST.SCOPE)
            lat=ZONE.centroid.y
            lon=ZONE.centroid.x
            D=_haversine_distances(list(zip(_torch.pi/180*lat,_torch.pi/180*lon)))
    if CITY.NAME=='NB':
        D_full=_torch.zeros((_CONST.NUM_LON*_CONST.NUM_LAT,_CONST.NUM_LON*_CONST.NUM_LAT))
        for a in range(_CONST.NUM_LON*_CONST.NUM_LAT):
            for b in range(_CONST.NUM_LON*_CONST.NUM_LAT):
                (a_x,a_y),(b_x,b_y)=N_to_XY(a),N_to_XY(b)
                D_full[a,b]=((a_x-b_x)**2+(a_y-b_y)**2)**0.5
        D=D_full[_CONST.CLIP,:][:,_CONST.CLIP]
    return _torch.tensor(D.max()-D).unsqueeze(-1).float()


def Random_Walk():
    A=_torch.tensor(_loader.Load_Adjacency(_CONST.SCOPE))
    M=A/A.sum(1,keepdim=True)
    return _torch.stack([M.matrix_power(i) for i in range(1,1+3)],-1).float()


def Point_of_Interest():
    return _torch.tensor(_loader.Load_POI(_CONST.SCOPE)).float()


def OD_Flow():
    return _torch.from_numpy(_loader.Load_Flow(_CONST.SCOPE)).float()
