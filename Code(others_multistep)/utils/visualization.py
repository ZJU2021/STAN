import warnings as _warnings
import CONST as _CONST
import numpy as _numpy
import pandas as _pandas
import matplotlib.pyplot as _plt
from utils import converter as _converter
from utils.loader import Load_Zone as _Load_Zone
from utils.loader import Load_Road as _Load_Road

_plt.rcParams['font.sans-serif']=['Times New Roman']
_plt.rcParams['axes.unicode_minus']=False
_plt.rcParams['text.usetex']=False



def Axes_empty():
    (fig,ax)=_plt.subplots(1,1,figsize=(6,4),dpi=120)
    fig.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.95)
    return ax


def Axes_year():
    ax=Axes_empty()
    ax.set_xlim(0,365*24)
    ax.set_xticks([24*(_pandas.to_datetime(f'2009{i:>02}01').day_of_year-1) for i in range(1,1+12)],['Jan.','Feb.','Mar.','Ari.','May.','Jun.','Jul.','Aut.','Sep.','Oct.','Nov.','Dec.'],verticalalignment='top',horizontalalignment='left')
    ax.grid(linestyle='dashed',alpha=_CONST.ALPHA_GRID,color='k')
    return ax


def Axes_week():
    ax=Axes_empty()
    ax.set_xlim(0,7*24)
    ax.set_xticks(range(0,7*24,24),['Mon.','Tue.','Wed.','Thu.','Fri.','Sat.','Sun.'],verticalalignment='top',horizontalalignment='left')
    ax.set_xticks(range(0,7*24,6),minor=True)
    ax.grid(which='major',linestyle='dashed',alpha=_CONST.ALPHA_GRID,color='k')
    ax.grid(which='minor',linestyle='dashed',alpha=0.5*_CONST.ALPHA_GRID,color='k')
    return ax


def Axes_day():
    ax=Axes_empty()
    ax.set_xlim(0,24)
    ax.set_xticks(range(0,24+1,3),[f'{i:>02}:00' for i in range(0,24+1,3)],verticalalignment='top',horizontalalignment='center')
    ax.set_xticks(range(0,24+1,1),minor=True)
    ax.grid(linestyle='dashed',alpha=_CONST.ALPHA_GRID,color='k')
    return ax


def DistShow(data,vmin=None,vmax=None,width=1):
    ax=Axes_empty()
    data_=_numpy.array(data)
    if vmin==None: vmin=_numpy.min(data)
    if vmax==None: vmax=_numpy.max(data)
    ticks=_numpy.arange(vmin,vmax,width)
    ticks=_numpy.concatenate((ticks,[ticks[-1]+width]))
    PDF=_numpy.diff([(data_<i).sum() for i in ticks])
    ax.bar(ticks[:-1]+width/2,PDF,width=width,color=_CONST.DEFAULT_COLOR)
    ax.set_xlim(vmin,vmax)
    ax.set_ylim(0,1.1*PDF.max())
    return ax


def Imshow(data,vmin=None,vmax=None,cmap=_CONST.DEFAULT_CMAP,ax=None):
    if ax==None: ax=Axes_empty()
    if vmin==None: vmin=_numpy.min(data)
    if vmax==None: vmax=_numpy.max(data)
    mappable=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal',extent=(0,_numpy.shape(data)[1],0,_numpy.shape(data)[0]))
    colorbar=_plt.colorbar(mappable,ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return (ax,colorbar)


def Map(size=1,is_road_on=True,is_zone_id_on=False):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        ax=_Load_Zone(_CONST.SCOPE).plot(ax=Axes_empty(),facecolor=(0,0,0,0.05),edgecolor=(0,0,0,1),linewidth=size*_CONST.DEFAULT_LINEWIDTH)
        if SCOPE=='full':
            ax.set_xticks(_numpy.arange(-74.2,-73.6,0.1))
            ax.set_yticks(_numpy.arange(40.5,41.0,0.1))
            ax.set_xlim(-74.28,-73.68)
            ax.set_ylim(40.48,40.93)
        if SCOPE=='clip':
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-74.04,-73.89)
            ax.set_ylim(40.69,40.89)
        ax.set_aspect(1/_numpy.cos(40.7/180*_numpy.pi))
        if is_road_on: ax=_Load_Road(_CONST.SCOPE).plot(ax=ax,facecolor=(0,0,0,0),edgecolor=(0,0,0,0.4),linewidth=0.4*_CONST.DEFAULT_LINEWIDTH)
        if is_zone_id_on:
            centroids=_converter.Centroids(ZONE)
            for zone_id in range(_CONST.NUM_ZONEs): ax.text(centroids[zone_id][0],centroids[zone_id][1],zone_id,size=5,color='k',verticalalignment='center',horizontalalignment='center')
    return ax


def Map_highlight(zone_ids,color=_CONST.DEFAULT_COLOR,ax=None):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        if ax==None: ax=Map()
        if isinstance(zone_ids,int): zone_ids=[zone_ids]
        zones=ZONE.loc[zone_ids]
        centroids=_converter.Centroids(ZONE)
        ax=zones.plot(ax=ax,facecolor=(color,0.6),edgecolor=(color,1),linewidth=2*_CONST.DEFAULT_LINEWIDTH)
        for zone_id in zone_ids: ax.text(centroids[zone_id][0],centroids[zone_id][1],zone_id,verticalalignment='center',horizontalalignment='center')
    return ax


def Map_value(data,vmin=None,vmax=None,cmap=_CONST.DEFAULT_CMAP,ax=None):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        if ax==None: ax=Map()
        if vmin==None: vmin=_numpy.min(data)
        if vmax==None: vmax=_numpy.max(data)
        ax=ZONE.plot(ax=ax,column=_numpy.array(data),cmap=cmap,vmin=vmin,vmax=vmax,edgecolor=(0,0,0,0),linewidth=0)
        temp=[str(type(x))=="<class 'matplotlib.collections.PatchCollection'>" for x in ax.get_children()]
        colorbar=_plt.colorbar(ax.get_children()[len(temp)-temp[::-1].index(True)-1],ax=ax)
    return (ax,colorbar)


def Map_radiation(zone_id,data,vmin=None,vmax=None,cmap=_CONST.DEFAULT_CMAP,size=1,ax=None):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        if ax==None: ax=Map()
        color=_converter.Value_to_Color(data,vmin,vmax,cmap)
        size=_converter.Value_to_Size(data,vmin,vmax,size)
        centroids=_converter.Centroids(ZONE)
        for i in range(_CONST.NUM_ZONEs):
            if data[i]>0: ax.plot(*zip(centroids[zone_id],centroids[i]),color=color[i],linewidth=size[i])
    return ax


def Map_OD(data,vmin=None,vmax=None,cmap=_CONST.DEFAULT_CMAP,size=1,ax=None):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        if ax==None: ax=Map()
        width=_numpy.shape(data)[1]
        data_=_numpy.flatten(data)
        color=Value_to_Color(data_,vmin,vmax,cmap)
        size=Value_to_Size(data_,vmin,vmax,size)
        centroids=_converter.Centroids(ZONE)
        for i in range(_CONST.NUM_ZONEs):
            for j in range(_CONST.NUM_ZONEs):
                index=i+j*width
                if data_[index]>0: ax.plot(*zip(centroids[i],centroids[j]),color=color[index],linewidth=size[index])
    return ax


def Plot_Shape(shape,color=_CONST.DEFAULT_COLOR,ax=None):
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        if ax==None: ax=Map()
        ax=shape.plot(ax=ax,facecolor=(color,0.6),edgecolor=(color,1),linewidth=2*_CONST.DEFAULT_LINEWIDTH)
    return ax


def Temporal_Correlation(lookback):
    flow=FLOW.sum(1).sum(1).reshape((365,_CONST.NUM_SLOTs_day)).T
    corr=_numpy.full((_CONST.NUM_SLOTs_day,2*_CONST.NUM_SLOTs_day),_numpy.nan)
    for i in range(_CONST.NUM_SLOTs_day):
        for j in range(_CONST.NUM_SLOTs_day):
            if j<i: corr[i,j+_CONST.NUM_SLOTs_day]=_numpy.corrcoef(flow[i][1:],flow[j][1:])[0,1]
            else: corr[i,j]=_numpy.corrcoef(flow[i][1:],flow[j][:-1])[0,1]

    (ax,colorbar)=Imshow(_numpy.flipud(corr),-1,1,'turbo')
    offset=_CONST.NUM_SLOTs_day-_converter.Hour_to_SlotID(lookback)
    for i in range(NUM_SLOTs_day):
        ax.fill((_CONST.NUM_SLOTs_day+i,_CONST.NUM_SLOTs_day+i,_CONST.NUM_SLOTs_day+i+1,_CONST.NUM_SLOTs_day+i+1),(i,i+1,i+1,i),facecolor='k',edgecolor=(0,0,0,0))
        ax.plot((offset+i-1,_CONST.NUM_SLOTs_day+i),(i,i),color='k',linewidth=2*_CONST.DEFAULT_LINEWIDTH)
        ax.plot((offset+i,offset+i),(i,i+1),color='k',linewidth=2*_CONST.DEFAULT_LINEWIDTH)
    ax.set_xticks(range(0,2*_CONST.NUM_SLOTs_day+1,_converter.Hour_to_SlotID(3)),list(range(0,24,3))+list(range(0,24+1,3)))
    ax.set_yticks(range(0,_CONST.NUM_SLOTs_day+1,_converter.Hour_to_SlotID(3)),range(0,24+1,3))
    ax.grid(linestyle='dashed',alpha=_CONST.ALPHA_GRID,color='k')
    return ax


def Compare_Predict_Target(predict,target):
    (fig,axs)=_plt.subplots(1,3,figsize=(9,4),dpi=100)
    fig.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95)
    
    which={'predict':0,'target':1,'diff':2}
    predict=predict.cpu().detach().numpy()
    target=target.cpu().detach().numpy()
    diff=predict-target
    (vmin,vmax)=(min(predict.min(),target.min()),max(predict.max(),target.max()))
    v=_numpy.abs(diff).max()
    
    mappable_predict=axs[which['predict']].imshow(predict,vmin=vmin,vmax=vmax,cmap='turbo',aspect='equal',extent=(0,predict.shape[1],0,predict.shape[0]))
    colorbar_predict=_plt.colorbar(mappable_predict,ax=axs[which['predict']],location='bottom',orientation='horizontal')
    
    mappable_target=axs[which['target']].imshow(target,vmin=vmin,vmax=vmax,cmap='turbo',aspect='equal',extent=(0,target.shape[1],0,target.shape[0]))
    colorbar_target=_plt.colorbar(mappable_target,ax=axs[which['target']],location='bottom',orientation='horizontal')
    
    mappable_diff=axs[which['diff']].imshow(diff,vmin=-v,vmax=v,cmap='turbo',aspect='equal',extent=(0,diff.shape[1],0,diff.shape[0]))
    colorbar_diff=_plt.colorbar(mappable_diff,ax=axs[which['diff']],location='bottom',orientation='horizontal')

    for i in range(len(axs)):
        axs[i].set_title(list(which.keys())[i])
        axs[i].set_xticks(range(1,diff.shape[1],8))
        axs[i].set_yticks(range(diff.shape[0]-1,0,-8),range(1,diff.shape[0],8))
        if i<2: axs[i].grid(color='w',alpha=_CONST.ALPHA_GRID)
        else: axs[i].grid(color='k',alpha=_CONST.ALPHA_GRID)
    
    return (fig,axs)