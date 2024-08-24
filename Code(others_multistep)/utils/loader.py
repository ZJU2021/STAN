import PATH as _PATH
import numpy as _numpy
import pandas as _pandas
import geopandas as _geopandas



def Load_Zone(scope):
    return _geopandas.read_file(_PATH.FOLDER_DATA+_PATH.FILE_ZONE[scope])

def Load_Road(scope,raw=False):
    if raw: return _geopandas.read_file(_PATH.FOLDER_DATA+_PATH.FILE_ROAD['raw'])
    else: return _geopandas.read_file(_PATH.FOLDER_DATA+_PATH.FILE_ROAD[scope])

def Load_Adjacency(scope):
    return _numpy.load(_PATH.FOLDER_DATA+_PATH.FILE_ADJACENCY[scope])

def Load_FsqCat(scope):
    return _pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_FSQCAT[scope])

def Load_Checkin(scope):
    return _pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_CHECKIN[scope])

def Load_POI(scope):
    return _numpy.load(_PATH.FOLDER_DATA+_PATH.FILE_POI[scope])

def Load_Weather(scope):
    return _pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_WEATHER[scope])

def Load_Trip(scope):
    TRIP=_pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP[scope])
    if scope=='full':
        TRIP['StartZone']-=1
        TRIP['EndZone']-=1
    return TRIP

def Load_Flow(scope):
    return _numpy.load(_PATH.FOLDER_DATA+_PATH.FILE_FLOW[scope])