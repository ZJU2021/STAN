import CITY
import numpy as _numpy
import pandas as _pandas



SCOPE='clip'
DEFAULT_CMAP='turbo'
DEFAULT_COLOR=(0.2,0.4,0.6)
DEFAULT_LINEWIDTH=0.2
ALPHA_GRID=0.25


if CITY.NAME=='NYC':
    SLOT=1800
    SLOT_to_Hour=SLOT/3600
    NUM_SLOTs_day=86400//SLOT
    NUM_SLOTs_week=7*NUM_SLOTs_day
    NUM_SLOTs_year=365*NUM_SLOTs_day

    TICKs_year=_numpy.arange(SLOT_to_Hour/2,365*24,SLOT_to_Hour)
    TICKs_week=_numpy.arange(SLOT_to_Hour/2,7*24,SLOT_to_Hour)
    TICKs_day=_numpy.arange(SLOT_to_Hour/2,24,SLOT_to_Hour)
    
    T_start=_pandas.to_datetime('2009-01-01 00:00:00')
    HOLIDAYs=[_pandas.to_datetime(x).day_of_year-1 for x in ('2009-01-01','2009-01-19','2009-02-16','2009-05-25','2009-07-03','2009-09-07','2009-10-12','2009-11-11','2009-11-26','2009-12-25')]
    MONTH_to_DIGIT={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

    NUM_CATEGORYs=9
    CATEGORYs=['Food','Shops','Building','Parks Outdoors','Arts Entertainment','Travel','Education','Nightlife','Home']
    HOMEs=['4e67e38e036454776db1fb3a','4bf58dd8d48988d103941735','52f2ab2ebcbc57f1066b8b55']

    CLIP=[3,11,12,23,40,41,42,44,47,49,67,73,74,78,86,87,89,99,106,112,113,115,119,124,126,127,136,139,140,141,142,143,147,150,151,152,157,160,161,162,163,165,169,185,193,201,208,210,223,228,229,230,231,232,233,235,236,237,238,242,243,245,248,260,261,262]
    FULL_to_CLIP={CLIP[i]:i for i in range(len(CLIP))}
    if SCOPE=='full': NUM_ZONEs=263
    if SCOPE=='clip': NUM_ZONEs=len(CLIP)

if CITY.NAME=='NB':
    SLOT=900
    SLOT_to_Hour=SLOT/3600
    NUM_SLOTs_day=86400//SLOT
    NUM_SLOTs_week=7*NUM_SLOTs_day
    NUM_SLOTs_year=365*NUM_SLOTs_day
    NUM_SLOTs=(31+1)*NUM_SLOTs_day
    T_start=_pandas.to_datetime('2019-12-01 00:00:00')
    
    NUM_CATEGORYs=20
    
    LON_MIN=121.412
    LON_MAX=121.754
    LAT_MIN=29.715
    LAT_MAX=30.038
    LON_UNIT=3*(0.009/_numpy.cos(_numpy.pi*30/180))
    LAT_UNIT=3*0.009
    NUM_LON=int(_numpy.ceil((LON_MAX-LON_MIN)/LON_UNIT))
    NUM_LAT=int(_numpy.ceil((LAT_MAX-LAT_MIN)/LAT_UNIT))
    CLIP=[14,15,16,24,25,26,27,28,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,55,56,57,58,59,60,61,62,63,67,68,69,70,71,72,73,74,75,78,79,80,81,82,83,84,85,86,87,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,115,116]
    FULL_to_CLIP={CLIP[i]:i for i in range(len(CLIP))}
    if SCOPE=='full': NUM_ZONEs=132
    if SCOPE=='clip': NUM_ZONEs=len(CLIP)
