import torch as _torch
from numpy.random import choice as _choice
from sklearn.linear_model import LinearRegression as _LR
from sklearn.ensemble import GradientBoostingRegressor as _GBR


def Get_Train_Test(dataset):
    X_train=_torch.stack([x[0][-1] for x in dataset['train']]).permute(0,2,3,1).flatten(0,2)
    y_train=_torch.stack([x[1] for x in dataset['train']]).flatten()

    X_test=_torch.stack([x[0][-1] for x in dataset['test']]).permute(0,2,3,1).flatten(0,2)
    y_test=_torch.stack([x[1] for x in dataset['test']]).flatten()

    sample=_choice(range(len(X_train)),2**22,False)
    return ((X_train[sample],y_train[sample]),(X_test,y_test))


def LR(dataset):
    ((X_train,y_train),(X_test,y_test))=Get_Train_Test(dataset)
    predict=_torch.tensor(_LR().fit(X_train,y_train).predict(X_test)).float()
    predict=predict.where(predict>0,0)
    target=y_test
    return (predict.reshape((len(dataset['test']),-1)),target.reshape((len(dataset['test']),-1)))

def GBR(dataset):
    ((X_train,y_train),(X_test,y_test))=Get_Train_Test(dataset)
    predict=_torch.tensor(_GBR().fit(X_train,y_train).predict(X_test)).float()
    predict=predict.where(predict>0,0)
    target=y_test
    return (predict.reshape((len(dataset['test']),-1)),target.reshape((len(dataset['test']),-1)))
