import CITY



FOLDER_MODEL='../Model'

if CITY.NAME=='NYC':
    FOLDER_DATA='../Data_NYC'
    FILE_ADJACENCY={'clip':'/Adjacency/clip/Adjacency.npy',
                    'full':'/Adjacency/full/Adjacency.npy',
                    'raw':None}
    FILE_CHECKIN={'clip':'/Checkin/clip/Checkin.csv',
                  'full':'/Checkin/full/Checkin.csv',
                  'raw':'/Checkin/raw/dataset_TSMC2014_NYC.txt'}
    FILE_FLOW={'clip':'/Flow/clip/Flow.npy',
               'full':'/Flow/full/Flow.npy',
               'raw':None}
    FILE_FSQCAT={'clip':'/FsqCat/clip/FsqCat.csv',
                 'full':'/FsqCat/full/FsqCat.csv',
                 'raw':'/FsqCat/raw/FsqCat.txt'}
    FILE_POI={'clip':'/POI/clip/POI.npy',
              'full':'/POI/full/POI.npy',
              'raw':None}
    FILE_ROAD={'clip':'/Map/clip/gis_osm_roads_free_1.shp',
               'full':'/Map/full/gis_osm_roads_free_1.shp',
               'raw':'/Map/raw/gis_osm_roads_free_1.shp'}
    FILE_SUBWAY={'clip':'/Subway/clip/Subway.csv',
                 'full':'/Subway/full/Subway.csv',
                 'raw':'/Subway/raw/geo_export_dda08828-cc99-4bd3-ba15-bd7b6e19eeaa.shp'}
    FILE_TRIP={'clip':'/Trip/clip/Trip.csv',
               'full':'/Trip/full/Trip.csv',
               'raw1':'/Trip/raw1', ##
               'raw2':'/Trip/raw2'} ##
    FILE_WEATHER={'clip':'/Weather/clip/Weather.csv',
                  'full':'/Weather/full/Weather.csv',
                  'raw':'/Weather/raw'} ##
    FILE_ZONE={'clip':'/Zone/clip/taxi_zones.shp',
               'full':'/Zone/full/taxi_zones.shp',
               'raw':'/Zone/raw/taxi_zones.shp'}

if CITY.NAME=='NB':
    FOLDER_DATA='../Data_NB'
    FILE_ADJACENCY={'clip':'/Adjacency/clip/Adjacency.npy'}
    FILE_POI={'clip':'/POI/clip/POI.npy',
              'raw':'/POI/raw/宁波市.csv'}
    FILE_FLOW={'clip':'/Flow/clip/Flow_abnormal.npy',
               'raw':'/Flow/raw/NBcombined.csv'}
    FILE_ROAD={'clip':'/Map/clip/ningbo.shp'}