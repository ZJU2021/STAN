import torch as _torch
from torch import nn as _nn



def MSELoss(predict,target): return (predict-target).pow(2).mean(0)

def MAELoss(predict,target): return (predict-target).abs().mean(0)
    
def MSE5Loss(predict,target,mask):
    diff=predict-target
    return _torch.tensor([diff[:,i][mask[:,i]].pow(2).mean() for i in range(diff.shape[1])])

def MAE5Loss(predict,target,mask):
    diff=predict-target
    return _torch.tensor([diff[:,i][mask[:,i]].abs().mean() for i in range(diff.shape[1])])

def MAPE5Loss(predict,target,mask):
    diff=predict-target
    return _torch.tensor([100*(diff[:,i][mask[:,i]]/target[:,i][mask[:,i]]).abs().mean() for i in range(diff.shape[1])])


class Metrics(_nn.Module):
    def __init__(self,flow_amp):
        super().__init__()
        self.flow_amp=flow_amp
        self.a=(flow_amp**2,flow_amp,flow_amp**2,flow_amp,1)
        
    def forward(self,predict,target):
        predict=predict.flatten(0,2)
        target=target.flatten(0,2)
        mask=target>5/self.flow_amp
        return [(self.a[0]*MSELoss(predict,target)).tolist(),
                (self.a[1]*MAELoss(predict,target)).tolist(),
                (self.a[2]*MSE5Loss(predict,target,mask)).tolist(),
                (self.a[3]*MAE5Loss(predict,target,mask)).tolist(),
                (self.a[4]*MAPE5Loss(predict,target,mask)).tolist()]


def Format_Metrics(metrics):
    (mse,mae,mse5,mae5,mape5)=metrics
    mse=' '.join([f'{x:5.3f}' for x in mse])
    mae=' '.join([f'{x:5.3f}' for x in mae])
    mse5=' '.join([f'{x:5.3f}' for x in mse5])
    mae5=' '.join([f'{x:5.3f}' for x in mae5])
    mape5=' '.join([f'{x:5.3f}%' for x in mape5])
    return f'MSE={mse}  MAE={mae}  MSE@5={mse5}  MAE@5={mae5}  MAPE@5={mape5}%'