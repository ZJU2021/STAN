import CONST as _CONST
import PARAM as _PARAM
from torch import stack as _stack



def HAP(FLOW_,k=3):  # History Average recent Periods
    return (_stack([FLOW_[_PARAM.VT_SPLIT-i*1: _PARAM.END-i*1] for i in range(1,1+k)]).mean(0),
            FLOW_[_PARAM.VT_SPLIT:_PARAM.END])

def HAD(FLOW_,k=3):  # History Average Days
    return (_stack([FLOW_[_PARAM.VT_SPLIT-i*_CONST.NUM_SLOTs_day:_PARAM.END-i*_CONST.NUM_SLOTs_day] for i in range(1,1+k)]).mean(0),
            FLOW_[_PARAM.VT_SPLIT:_PARAM.END])

def HAW(FLOW_,k=3):  # History Average Weeks
     return (_stack([FLOW_[_PARAM.VT_SPLIT-i*_CONST.NUM_SLOTs_week:_PARAM.END-i*_CONST.NUM_SLOTs_week] for i in range(1,1+k)]).mean(0),
            FLOW_[_PARAM.VT_SPLIT:_PARAM.END])
