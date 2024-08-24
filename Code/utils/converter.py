import warnings as _warnings
import CONST as _CONST
import numpy as _numpy
import pandas as _pandas
from shapely.geometry import Point as _Point
from scipy.signal import convolve as _convolve
from matplotlib.colors import Normalize as _Normalize
from matplotlib.cm import ScalarMappable as _ScalarMappable



def XY_to_N(x,y,first='row'):
    if first=='row': return _CONST.NUM_LON*y+x
    if first=='column': return _CONST.NUM_LAT*x+y

def N_to_XY(n,first='row'):
    if first=='row': return (n%_CONST.NUM_LON,n//_CONST.NUM_LON)
    if first=='column': return (n%_CONST.NUM_LAT,n//_CONST.NUM_LAT)


def Smooth(data,n=1,cyclical=False):
    data_=_numpy.array(data)
    if cyclical:
        for i in range(n): data_=_convolve(_numpy.concatenate([[data_[-1]],data_,[data_[0]]]),(0.25,0.5,0.25),mode='valid')
    else:
        for i in range(n): data_=_convolve(_numpy.concatenate([[data_[0]],data_,[data_[-1]]]),(0.25,0.5,0.25),mode='valid')
    return data_


def Reshape(n,k,offset=0):
    a=[[i+j*k for j in range(n//k+1)] for i in range(n%k)]+[[i+j*k for j in range(n//k)] for i in range(n%k,k)]
    return a[k-offset:]+a[:k-offset]


def Splitter(n,k):
    p=_numpy.floor(_numpy.linspace(0,n,k,endpoint=False)).astype(int).tolist()+[n]
    return list(zip(p[:-1],p[1:]))


def Value_to_Color(data,vmin=None,vmax=None,cmap='turbo'):
    if vmin==None: vmin=_numpy.min(data)
    if vmax==None: vmax=_numpy.max(data)
    SM=_ScalarMappable(_Normalize(vmin,vmax),cmap).to_rgba
    return [(*SM(value)[:3],_numpy.clip((value-vmin)/(vmax-vmin+1e-6),0,1)) for value in _numpy.flatten(data)]


def Value_to_Size(data,vmin=None,vmax=None,size=1):
    if vmin==None: vmin=_numpy.min(data)
    if vmax==None: vmax=_numpy.max(data)
    return [size*_numpy.clip((value-vmin)/(vmax-vmin+1e-6),0,1) for value in _numpy.flatten(data)]


def Norm_MinMax(X,axis=None,need_minmax=False,need_amplitude=False):
    if axis==None:
        X_min=X.min().item()
        X_max=X.max().item()
    else:
        if isinstance(axis,int): axis=[axis]
        X_min=X
        X_max=X
        for i in axis:
            X_min=X_min.min(i,keepdim=True)[0]
            X_max=X_max.max(i,keepdim=True)[0]
    if need_minmax: return ((X-X_min)/(X_max-X_min+1e-9),(X_min,X_max))
    elif need_amplitude: return ((X-X_min)/(X_max-X_min+1e-9),X_max-X_min)
    else: return (X-X_min)/(X_max-X_min+1e-9)


def LonLat_to_ZoneID(lon,lat,ZONE):
    for index in ZONE.index:
        if ZONE.at[index,'geometry'].contains(_Point((lon,lat))): return index
    return -1


def Centroids(ZONE):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        centroids=list(zip(ZONE.centroid.x,ZONE.centroid.y))
    return centroids


def Neighbours(ADJACENCY):
    edge=_numpy.argwhere(ADJACENCY)
    return [edge[edge[:,0]==zone_id][:,1].tolist() for zone_id in range(len(ADJACENCY))]


def Format_Timestamp(timestamp):
    x=timestamp.split(' ')
    return f'{x[5]}-{_CONST.MONTH_to_DIGIT[x[1]]}-{x[2]} {x[3]}'


def Timestamp_to_Second(timestamp):
    return (_pandas.to_datetime(timestamp)-_CONST.T_start).value//1000000000


def Second_to_SlotID(seconds):
    return seconds//_CONST.SLOT


def Hour_to_SlotID(hour):
    return int(3600*hour//_CONST.SLOT)
