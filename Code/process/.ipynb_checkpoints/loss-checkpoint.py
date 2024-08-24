from torch import nn as _nn



def MSELoss(predict,target): return (predict-target).pow(2).mean()

def MAELoss(predict,target): return (predict-target).abs().mean()
    
def MSE5Loss(predict,target,mask): return (predict-target)[mask].pow(2).mean()

def MAE5Loss(predict,target,mask): return (predict-target)[mask].abs().mean()
    
def MAPE5Loss(predict,target,mask): return 100*((predict-target)[mask]/target[mask]).abs().mean()


class Metrics(_nn.Module):
    def __init__(self,flow_amp):
        super().__init__()
        self.flow_amp=flow_amp
        self.a=(flow_amp**2,flow_amp,flow_amp**2,flow_amp,1)
        
    def forward(self,predict,target):
        mask=target>5/self.flow_amp
        return (self.a[0]*MSELoss(predict,target).item(),
                self.a[1]*MAELoss(predict,target).item(),
                self.a[2]*MSE5Loss(predict,target,mask).item(),
                self.a[3]*MAE5Loss(predict,target,mask).item(),
                self.a[4]*MAPE5Loss(predict,target,mask).item())


def Format_Metrics(metrics):
    (mse,mae,mse5,mae5,mape5)=metrics
    return f'MSE={mse:5.3f} MAE={mae:5.3f} MSE@5={mse5:5.3f} MAE@5={mae5:5.3f} MAPE@5={mape5:5.3f}%'