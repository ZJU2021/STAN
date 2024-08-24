import CITY
import warnings as _warnings
import CONST as _CONST
import PATH as _PATH
import numpy as _numpy
import pandas as _pandas
import geopandas as _geopandas
from utils import loader as _loader
from utils import converter as _converter
from os import listdir as _listdir
from json import loads as _loads
from tqdm import tqdm as _tqdm
from pyarrow.parquet import read_table as _read_table


if CITY.NAME=='NYC':
    def Make_Zone(scope):
        if scope=='full':
            ZONE=_geopandas.read_file(_PATH.FOLDER_DATA+_PATH.FILE_ZONE['raw']).to_crs("EPSG:4326").drop(columns=['LocationID','Shape_Leng','OBJECTID','Shape_Area']).rename(columns={'zone':'Name','borough':'Borough'})
            ZONE.to_file(_PATH.FOLDER_DATA+_PATH.FILE_ZONE['full'])

        ZONE=_loader.Load_Zone('full')
        ZONE.loc[_CONST.CLIP,:].reset_index(drop=True).to_file(_PATH.FOLDER_DATA+_PATH.FILE_ZONE['clip'])
        print('Finish making ZONE')


    def Make_Road(scope):
        if scope=='full':
            ZONE=_loader.Load_Zone('full')
            ROAD=_geopandas.read_file(_PATH.FOLDER_DATA+_PATH.FILE_ROAD['raw'])
            ROAD.clip(ZONE)[['geometry']].to_file(_PATH.FOLDER_DATA+_PATH.FILE_ROAD['full'])

        ROAD=_loader.Load_Road('full')
        ROAD.clip(_loader.Load_Zone('full').loc[_CONST.CLIP,:])[['geometry']].to_file(_PATH.FOLDER_DATA+_PATH.FILE_ROAD['clip'])
        print('Finish making ROAD')


    def Make_Adjacency(scope,err=0):
        if scope=='full':
            ZONE=_loader.Load_Zone('full')
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                ADJACENCY=_numpy.zeros((len(ZONE),len(ZONE)))
                for i in range(len(ZONE)):
                    for j in range(i):
                        if err>0:
                            if (ZONE.loc[[i]].buffer(err)).intersects((ZONE.loc[[j]].buffer(err)),align=False).iat[0]: ADJACENCY[i,j]=1
                        else:
                            if ZONE.loc[[i]].intersects(ZONE.loc[[j]],align=False).iat[0]: ADJACENCY[i,j]=1
            ADJACENCY+=ADJACENCY.T

            ADJACENCY_add=[(201,228),
                           (140,201),
                           (192,201),
                           (144,201),
                           (145,201),
                           (44,64),
                           (44,65),
                           (64,208),
                           (65,208),
                           (64,147),
                           (65,147),
                           (12,104),
                           (104,260),
                           (104,194),
                           (39,104),
                           (147,255),
                           (231,255),
                           (144,169),
                           (167,193),
                           (178,193),
                           (193,222),
                           (198,222),
                           (207,251),
                           (14,207),
                           (5,13),
                           (13,220),
                           (26,153),
                           (29,116),
                           (29,200),
                           (85,202),
                           (1,123),
                           (111,144),
                           (111,225),
                           (79,225),
                           (127,219),
                           (127,152),
                           (126,152),
                           (126,135),
                           (126,234),
                           (234,243),
                           (118,243),
                           (41,118),
                           (41,246),
                           (73,246),
                           (73,167),
                           (74,193),
                           (45,183)]
            ADJACENCY_remove=[(137,198),
                              (52,137),
                              (92,137),
                              (131,202),
                              (1,131),
                              (43,98)]
            for (i,j) in ADJACENCY_add:
                ADJACENCY[i,j]=1
                ADJACENCY[j,i]=1
            for (i,j) in ADJACENCY_remove:
                ADJACENCY[i,j]=0
                ADJACENCY[j,i]=0

            _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_ADJACENCY['full'],ADJACENCY)

        ADJACENCY=_loader.Load_Adjacency('full')
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_ADJACENCY['clip'],ADJACENCY[_CONST.CLIP,:][:,_CONST.CLIP])
        print('Finish making ADJACENCY')


    def Make_FsqCat(scope):
        if scope=='full':
            data=[[y.strip("\'") for y in x.lstrip('(').rstrip(')').split(', ')] for x in open(_PATH.FOLDER_DATA+_PATH.FILE_FSQCAT['raw']).read().strip().split('\n')]
            FSQCAT=_pandas.DataFrame(data[1:],columns=data[0])[['parent_id','foursquare_id','icon_prefix']].rename(columns={'foursquare_id':'ItemCode','parent_id':'CategoryCode','icon_prefix':'String'})
            FSQCAT['CategoryName']=FSQCAT['String'].transform(lambda x: ' '.join([i.capitalize() for i in x.strip('_').split('/')[0].split('_')]))
            FSQCAT['ItemName']=FSQCAT['String'].transform(lambda x: ' '.join([i.capitalize() for i in x.strip('_').split('/')[1].split('_')]))
            FSQCAT.drop(columns=['String'],inplace=True)
            categories=FSQCAT['CategoryName'].value_counts().index.tolist()
            categories_to_id={categories[i]:i for i in range(len(categories))}
            FSQCAT['CategoryID']=FSQCAT['CategoryName'].transform(lambda x: categories_to_id[x])
            FSQCAT.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_FSQCAT['full'],index=False)

        FSQCAT=_loader.Load_FsqCat('full')
        FSQCAT.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_FSQCAT['clip'],index=False)
        print('Finish making FSQCAT')


    def Make_Checkin(scope):
        if scope=='full':
            ZONE=_loader.Load_Zone('full')
            FSQCAT=_loader.Load_FsqCat('full')
            CHECKIN=_pandas.DataFrame([row.split('\t') for row in open(_PATH.FOLDER_DATA+_PATH.FILE_CHECKIN['raw'],encoding='iso-8859-1').read().strip().split('\n')],columns=['UserID','VenueCode','ItemCode','ItemName','Lat','Lon','Offset','UTC'])
            CHECKIN['Lat']=CHECKIN['Lat'].astype(float)
            CHECKIN['Lon']=CHECKIN['Lon'].astype(float)
            CHECKIN['Offset']=_pandas.to_timedelta(CHECKIN['Offset']+' min')
            CHECKIN['UTC']=_pandas.to_datetime(CHECKIN['UTC'].transform(_converter.Format_Timestamp))
            CHECKIN['SlotID']=(CHECKIN['UTC']+CHECKIN['Offset']).transform(lambda x: 86400*x.day_of_week+3600*x.hour+60*x.minute+x.second).transform(_converter.Second_to_SlotID)
            CHECKIN['ZoneID']=CHECKIN.apply(lambda x: _converter.LonLat_to_ZoneID(x['Lon'],x['Lat'],ZONE),axis=1)
            CHECKIN.drop(index=CHECKIN[CHECKIN['ZoneID']<0].index,inplace=True)
            itemcode_to_categoryid={FSQCAT['ItemCode'][i]:FSQCAT['CategoryID'][i] for i in range(len(FSQCAT))}
            CHECKIN.drop(CHECKIN[~CHECKIN['ItemCode'].isin(list(itemcode_to_categoryid.keys()))].index,inplace=True)
            CHECKIN['CategoryID']=CHECKIN['ItemCode'].transform(lambda x: itemcode_to_categoryid[x])
            CHECKIN.loc[CHECKIN['ItemCode'].isin(_CONST.HOMEs),'CategoryID']=8
            CHECKIN.drop(columns=['UserID','VenueCode','ItemName','Offset','UTC','ItemCode','Lon','Lat'],inplace=True)
            CHECKIN.sort_values('SlotID',inplace=True)
            CHECKIN.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_CHECKIN['full'],index=False)

        CHECKIN=_loader.Load_Checkin('full')
        CHECKIN=CHECKIN[CHECKIN['ZoneID'].isin(_CONST.CLIP)]
        CHECKIN['ZoneID']=CHECKIN['ZoneID'].transform(lambda x: _CONST.FULL_to_CLIP[x])
        CHECKIN.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_CHECKIN['clip'],index=False)
        print('Finish making CHECKIN')


    def Make_POI(scope):
        if scope=='full':
            ZONE=_loader.Load_Zone('full')
            CHECKIN=_loader.Load_Checkin('full')
            POI=_numpy.zeros((len(ZONE),_CONST.NUM_CATEGORYs))
            count=CHECKIN[['ZoneID','CategoryID']].groupby(['ZoneID','CategoryID']).aggregate(len)
            for (index,n) in zip(count.index,count): POI[index]=n
            _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_POI['full'],POI)

        POI=_loader.Load_POI('full')
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_POI['clip'],POI[_CONST.CLIP,:])
        print('Finish making POI')


    def Make_Weather(scope):
        if scope=='full':
            files=[_PATH.FOLDER_DATA+_PATH.FILE_WEATHER['raw']+file for file in _listdir(_PATH.FOLDER_DATA+_PATH.FILE_WEATHER['raw'])]
            WEATHER_raw=_pandas.concat([_pandas.DataFrame().from_dict(_loads(open(file).read())['observations'])[['valid_time_gmt','temp','precip_hrly']] for file in files])
            WEATHER_raw.rename(columns={'valid_time_gmt':'Time','temp':'Temperature','precip_hrly':'Precipitation'},inplace=True)
            WEATHER_raw.drop(index=WEATHER_raw[(WEATHER_raw['Temperature'].isna())&(WEATHER_raw['Precipitation'].isna())].index,inplace=True)
            WEATHER_raw.fillna(0,inplace=True)
            WEATHER_raw['Time']=WEATHER_raw['Time']-5*3600-_pandas.to_datetime('2009-01-01 00:00:00').value//1000000000
            WEATHER_raw=WEATHER_raw.groupby('Time').aggregate(_numpy.mean)
            WEATHER_raw.sort_index(inplace=True)
            WEATHER=_pandas.DataFrame()
            WEATHER['SlotID']=range(_CONST.NUM_SLOTs_year)
            WEATHER['Temperature']=_numpy.interp(_CONST.SLOT*(0.5+WEATHER['SlotID']),WEATHER_raw.index,WEATHER_raw['Temperature'])
            WEATHER['Precipitation']=_numpy.interp(_CONST.SLOT*(0.5+WEATHER['SlotID']),WEATHER_raw.index,WEATHER_raw['Precipitation'])
            WEATHER.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_WEATHER['full'],index=False)

        WEATHER=_loader.Load_Weather('full')
        WEATHER.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_WEATHER['clip'],index=False)
        print('Finish making WEATHER')


    # import multiprocessing as mp

    # def Make_Trip_step1():
    #     TRIP=_pandas.DataFrame()
    #     for month in _tqdm(range(1,1+12),ncols=50):
    #         TRIP=_pandas.concat((TRIP,_read_table(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw1']+f'yellow_tripdata_2009-{month:>02}.parquet').to_pandas()[['Trip_Pickup_DateTime','Trip_Dropoff_DateTime','Start_Lon','Start_Lat','End_Lon','End_Lat','Trip_Distance']]))
    #     TRIP.rename(columns={'Trip_Pickup_DateTime':'StartTim',
    #                           'Trip_Dropoff_DateTime':'EndTim',
    #                           'Start_Lon':'StartLon',
    #                           'Start_Lat':'StartLat',
    #                           'End_Lon':'EndLon',
    #                           'End_Lat':'EndLat',
    #                           'Trip_Distance':'TripDist'},inplace=True)
    #     return TRIP

    # def Make_Trip_step2(TRIP_i,ZONE,process_id):
    #     ZONE=_loader.Load_Zone('full')
    #     for (block_id,(head,tail)) in enumerate(_tqdm(Splitter(len(TRIP_i),100),ncols=50)):
    #         TRIP_i_j=TRIP_i.iloc[head:tail].copy()
    #         TRIP_i_j['StartTime']=TRIP_i_j['StartTim'].transform(Timestamp_to_Second).transform(Second_to_SlotID)
    #         TRIP_i_j['EndTime']=TRIP_i_j['EndTim'].transform(Timestamp_to_Second).transform(Second_to_SlotID)
    #         TRIP_i_j['StartZone']=TRIP_i_j.apply(lambda x: _converter.LonLat_to_ZoneID(x['StartLon'],x['StartLat'],ZONE),axis=1)
    #         TRIP_i_j['EndZone']=TRIP_i_j.apply(lambda x: _converter.LonLat_to_ZoneID(x['EndLon'],x['EndLat'],ZONE),axis=1)
    #         invalid_time=TRIP_i_j[(TRIP_i_j['StartTime']<0)|(TRIP_i_j['StartTime']>NUM_SLOTs_year-1)|(TRIP_i_j['EndTime']<0)|(TRIP_i_j['EndTime']>NUM_SLOTs_year-1)].index.tolist()
    #         invalid_zone=TRIP_i_j[(TRIP_i_j['StartZone']<0)|(TRIP_i_j['EndZone']<0)].index.tolist()
    #         TRIP_i_j.drop(index=invalid_time+invalid_zone,inplace=True)
    #         TRIP_i_j.drop(columns=['StartTim','EndTim','StartLon','StartLat','EndLon','EndLat'],inplace=True)
    #         TRIP_i_j.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw2']+f'P{process_id:>02}_C{1+block_id:>03}__{len(TRIP_i_j)}.csv',index=False)

    # # if __name__=="__main__":
    # #     TRIP=Make_Trip_step1()
    # #     chunks=Splitter(len(TRIP),NUM_CPNUM_CPUs)
    # #     P=[mp.Process(target=Make_Trip_step2,args=[TRIP.iloc[chunks[i][0]:chunks[i][1]].copy(),ZONE,i+1]) for i in range(NUM_CPUs-1)]
    # #     for p in P: p.start()
    # #     for p in P: p.join()
    # #     print('Done!')
    # #     os.system('pause')

    # def Make_Trip_step3():
    #     TRIP=_pandas.concat([_pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw2']+file) for file in _listdir(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw2'])])
    #     TRIP.sort_values(['StartTime','EndTime'],inplace=True)
    #     TRIP.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['full'],index=False)
    #     TRIP[TRIP['StartZone'].isin(_CONST.CLIP)&TRIP['EndZone'].isin(_CONST.CLIP)].to_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['clip'],index=False)


    def Make_Trip(scope):
        if scope=='full':
            TRIP=_pandas.concat([_pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw2']+file) for file in _listdir(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['raw2'])])
            TRIP.sort_values(['StartTime','EndTime'],inplace=True)
            TRIP.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['full'],index=False)

        TRIP=_loader.Load_Trip('full')
        TRIP=TRIP[TRIP['StartZone'].isin(_CONST.CLIP)&TRIP['EndZone'].isin(_CONST.CLIP)]
        TRIP['StartZone']=TRIP['StartZone'].transform(lambda x: _CONST.FULL_to_CLIP[x])
        TRIP['EndZone']=TRIP['EndZone'].transform(lambda x: _CONST.FULL_to_CLIP[x])
        TRIP.to_csv(_PATH.FOLDER_DATA+_PATH.FILE_TRIP['clip'],index=False)
        print('Finish making TRIP')


    def Make_Flow(scope):
        if scope=='full':
            ZONE=_loader.Load_Zone('full')
            TRIP=_loader.Load_Trip('full')
            FLOW=_numpy.zeros((_CONST.NUM_SLOTs_year,len(ZONE),len(ZONE)),_numpy.float32)
            grouped=TRIP.groupby(['StartZone','EndZone'])
            for (i,j) in TRIP[['StartZone','EndZone']].value_counts().index:
                count=grouped.get_group((i,j))['StartTime'].value_counts()
                for (t,n) in zip(count.index,count): FLOW[t,i,j]=n
            _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_FLOW['full'],FLOW)

        FLOW=_loader.Load_Flow('full')
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_FLOW['clip'],FLOW[:,_CONST.CLIP,:][:,:,_CONST.CLIP])
        print('Finish making FLOW')



if CITY.NAME=='NB':
    def Make_Adjacency():
        ADJACENCY=_numpy.zeros((_CONST.NUM_LON*_CONST.NUM_LAT,_CONST.NUM_LON*_CONST.NUM_LAT))
        for a in range(_CONST.NUM_LON*_CONST.NUM_LAT):
            for b in range(_CONST.NUM_LON*_CONST.NUM_LAT):
                (a_x,a_y),(b_x,b_y)=_converter.N_to_XY(a),_converter.N_to_XY(b)
                if a_x==b_x and a_y==b_y: ADJACENCY[a,b]=0
                if a_x==b_x and abs(a_y-b_y)==1 or a_y==b_y and abs(a_x-b_x)==1: ADJACENCY[a,b]=1
                if abs(a_x-b_x)==1 and abs(a_y-b_y)==1: ADJACENCY[a,b]=1/4
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_ADJACENCY['clip'],ADJACENCY[_CONST.CLIP,:][:,_CONST.CLIP])

        
    def Make_POI():
        CHECKIN=_pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_POI['raw'],index_col=0)[['typecode','lon','lat']]
        CHECKIN['x']=(CHECKIN['lon'].astype(float)-_CONST.LON_MIN)/_CONST.LON_UNIT
        CHECKIN['y']=(CHECKIN['lat'].astype(float)-_CONST.LAT_MIN)/_CONST.LAT_UNIT
        CHECKIN['x']=CHECKIN['x'].astype(int)
        CHECKIN['y']=CHECKIN['y'].astype(int)
        CHECKIN.drop(columns=['lon','lat'],inplace=True)
        CHECKIN.drop(index=CHECKIN[(CHECKIN['x']<0)|(CHECKIN['x']>=_CONST.NUM_LON)|(CHECKIN['y']<0)|(CHECKIN['y']>=_CONST.NUM_LAT)].index,inplace=True)
        CHECKIN.reset_index(drop=True,inplace=True)

        POI=_numpy.zeros((_CONST.NUM_LON*_CONST.NUM_LAT,_CONST.NUM_CATEGORYs))
        for i in CHECKIN.index:
            for c in CHECKIN.loc[i,'typecode'].split('|'):
                if int(c[:2])<=_CONST.NUM_CATEGORYs: POI[_converter.XY_to_N(CHECKIN.loc[i,'x'],CHECKIN.loc[i,'y']),int(c[:2])-1]+=1
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_POI['clip'],POI[_CONST.CLIP])
    
    
    def Make_Flow():
        TRIP=_pandas.read_csv(_PATH.FOLDER_DATA+_PATH.FILE_FLOW['raw'])
        TRIP['O_x']=(TRIP['depLocationText'].transform(lambda x: x.split(',')[0]).astype(float)-_CONST.LON_MIN)/_CONST.LON_UNIT
        TRIP['O_y']=(TRIP['depLocationText'].transform(lambda x: x.split(',')[1]).astype(float)-_CONST.LAT_MIN)/_CONST.LAT_UNIT
        TRIP['D_x']=(TRIP['destLocationText'].transform(lambda x: x.split(',')[0]).astype(float)-_CONST.LON_MIN)/_CONST.LON_UNIT
        TRIP['D_y']=(TRIP['destLocationText'].transform(lambda x: x.split(',')[1]).astype(float)-_CONST.LAT_MIN)/_CONST.LAT_UNIT
        TRIP['T']=(_pandas.to_datetime(TRIP['depTime'].transform(lambda x: x[:-5]))+_pandas.to_timedelta('8h')-_CONST.T_start).transform(lambda x: x.value/1000000000)/_CONST.SLOT
        TRIP['O_x']=TRIP['O_x'].astype(int)
        TRIP['O_y']=TRIP['O_y'].astype(int)
        TRIP['D_x']=TRIP['D_x'].astype(int)
        TRIP['D_y']=TRIP['D_y'].astype(int)
        TRIP['O']=_CONST.NUM_LON*TRIP['O_y']+TRIP['O_x']
        TRIP['D']=_CONST.NUM_LON*TRIP['D_y']+TRIP['D_x']
        TRIP['T']=TRIP['T'].astype(int)
        TRIP.drop(columns=['depLocationText','destLocationText','depTime'],inplace=True)
        TRIP.drop(index=TRIP[(TRIP['O_x']<0)|(TRIP['O_x']>=_CONST.NUM_LON)|(TRIP['O_y']<0)|(TRIP['O_y']>=_CONST.NUM_LAT)|\
                             (TRIP['D_x']<0)|(TRIP['D_x']>=_CONST.NUM_LON)|(TRIP['D_y']<0)|(TRIP['D_y']>=_CONST.NUM_LAT)|\
                             (TRIP['T']<0)|(TRIP['T']>=_CONST.NUM_SLOTs)].index,inplace=True)
        TRIP.reset_index(drop=True,inplace=True)

        FLOW=_numpy.zeros((_CONST.NUM_SLOTs,_CONST.NUM_LON*_CONST.NUM_LAT,_CONST.NUM_LON*_CONST.NUM_LAT))
        count=TRIP[['T','O','D']].groupby(['T','O','D']).value_counts()
        for index in count.index: FLOW[index]=count[index]
        _numpy.save(_PATH.FOLDER_DATA+_PATH.FILE_FLOW['clip'],FLOW[:,_CONST.CLIP,:][:,:,_CONST.CLIP])