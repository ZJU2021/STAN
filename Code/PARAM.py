import CITY
from utils.converter import Hour_to_SlotID as _Hour_to_SlotID
from utils.converter import Timestamp_to_Second as _OP1
from utils.converter import Second_to_SlotID as _OP2



idx={'NYC':0,'NB':1}[CITY.NAME]
DEVICE='cuda'

START=_OP2(_OP1(['2009-02-01 00:00:00','2019-12-01 00:00:00'][idx]))
TV_SPLIT=_OP2(_OP1(['2009-10-01 00:00:00','2019-12-24 00:00:00'][idx]))
VT_SPLIT=_OP2(_OP1(['2009-11-01 00:00:00','2019-12-27 00:00:00'][idx]))
END=_OP2(_OP1(['2009-12-01 00:00:00','2019-12-31 00:00:00'][idx]))

NUM_HISTORY=_Hour_to_SlotID(16)
W_D_P=[(3,3,3),(0,0,6)][idx]

DIM_GLOBAL=8  # 8
DIM_EDGE=32  # 32
NUM_HEADs=4  # 4
NUM_BLOCKs=3  # 3
