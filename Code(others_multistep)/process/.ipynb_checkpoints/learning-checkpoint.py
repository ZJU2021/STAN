import CITY
import PARAM as _PARAM
import PATH as _PATH
import torch as _torch
import os as _os
from time import time as _get_time
from process.loss import Format_Metrics as _Format_Metrics
from torch.utils.data import DataLoader as _DataLoader
from torchinfo import summary as _summary



def Train(model,optimizer,dataloader,loss_fn,metrics_fn,k):
    model.train()
    n=len(dataloader)
    metrics=[]
    t_start=_get_time()
    
    for (batch,data) in enumerate(dataloader):
        (input,target)=data
        predict=model([x.to(_PARAM.DEVICE) for x in input])
        target=target.to(_PARAM.DEVICE)
        loss_fn(predict,target).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.append(metrics_fn(predict,target))
        
        if (1+batch)%k==0:
            t_end=_get_time()
            t=(t_end-t_start)/k
            t_start=t_end
            print(f'{1+batch:>5d}/{n} | {t:4.2f}s/B {t*k:2.0f}s/R {t*n/60:2.0f}m/E [{t*(n-batch-1)/60:2.0f}m] | '+\
                  _Format_Metrics([sum(x[-k:])/k for x in zip(*metrics)]))

    return [sum(x)/n for x in zip(*metrics)]


def Test(model,dataloader,metrics_fn):
    model.eval()
    n=len(dataloader)
    metrics=[]

    with _torch.inference_mode():
        for data in dataloader:
            (input,target)=data
            predict=model([x.to(_PARAM.DEVICE) for x in input])
            target=target.to(_PARAM.DEVICE)
            metrics.append(metrics_fn(predict.where(predict>0,0),target))
    
    return [sum(x)/n for x in zip(*metrics)]


def Loop(model,optimizer,scheduler,dataset,loss_fn,metrics_fn,
         batch_size=32,num_epochs=140,num_workers=0,k=100,load_checkpoint=False):
    print(str(_summary(model,depth=0)).splitlines()[-4]+'\n')

    dataloader={'train':_DataLoader(dataset['train'],batch_size,pin_memory=True,num_workers=num_workers,shuffle=True),
                'validate':_DataLoader(dataset['validate'],batch_size,pin_memory=True,num_workers=num_workers),
                'test':_DataLoader(dataset['test'],batch_size,pin_memory=True,num_workers=num_workers)}
    model=model.to(_PARAM.DEVICE)
    name=str(model.__class__).rstrip('\'>').split('.')[-1]
    path=_PATH.FOLDER_MODEL+f'/{CITY.NAME}/'+name
    if not _os.path.exists(path): _os.mkdir(path)
    start_epoch=1
    
    if load_checkpoint:
        if _os.listdir(path):
            last_file=path+'/'+sorted(_os.listdir(path))[-1]
            checkpoint=_torch.load(last_file)
            start_epoch+=checkpoint['epoch']
            metrics=checkpoint['metrics']
            model.load_state_dict(checkpoint['model_state_dict'])
            lr=optimizer.param_groups[0]["lr"]
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]["lr"]=lr
            del checkpoint
            print(last_file+'\n')
    
    for epoch in range(start_epoch,1+num_epochs):
        print(f'Epoch: {epoch:0>3d}/{num_epochs:0>3d}    Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        metrics={'train':Train(model,optimizer,dataloader['train'],loss_fn,metrics_fn,k),
                 'validate':Test(model,dataloader['validate'],metrics_fn),
                 'test':Test(model,dataloader['test'],metrics_fn)}
        scheduler.step()
 
        _torch.save({'epoch':epoch,'metrics':metrics,
                     'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'parameters':{'DIM_GLOBAL':_PARAM.DIM_GLOBAL,
                                   'DIM_EDGE':_PARAM.DIM_EDGE,
                                   'NUM_HEADs':_PARAM.NUM_HEADs,
                                   'NUM_BLOCKs':_PARAM.NUM_BLOCKs} if name=='Mine' else 0},
                    path+'/'+f'{name}_{int(_get_time())}_E{epoch:0>3d}_'+\
                    '_train_'+'_'.join([f'{x:5.3f}' for x in metrics["train"]])+\
                    '_validate_'+'_'.join([f'{x:5.3f}' for x in metrics["validate"]])+\
                    '_test_'+'_'.join([f'{x:5.3f}' for x in metrics["test"]])+'.pt')

        print(f'   Train Loss:  {_Format_Metrics(metrics["train"])}\n'+\
              f'Validate Loss:  {_Format_Metrics(metrics["validate"])}\n'+\
              f'    Test Loss:  {_Format_Metrics(metrics["test"])}\n')

        
def Test_multistep(model,dataset,metrics_fn,num_workers=0):
    model.eval()
    num_samples=len(dataset['test'])
    (input,target)=next(iter(_DataLoader(dataset['test'],batch_size=num_samples,pin_memory=True,num_workers=num_workers)))
    input=[x.to(_PARAM.DEVICE) for x in input]
    length=len(target)-(_PARAM.NUM_STEPs-1)
    target=target.to(_PARAM.DEVICE)
    batch_size=32
    predicts=[]

    with _torch.inference_mode():
        for step in range(_PARAM.NUM_STEPs):
            input_step_k=[x[step:num_samples-_PARAM.NUM_STEPs+step+1].clone() for x in input]
            target_step_k=target[step:num_samples-_PARAM.NUM_STEPs+step+1]
            for k in range(step): input_step_k[-1][:,-step+k]=predicts[k]
            
            temp=_torch.concatenate([model([x[i:i+batch_size] for x in input_step_k]) for i in range(0,length,batch_size)],0)
            predict_step_k=temp.where(temp>0,0).to(_PARAM.DEVICE)
            predicts.append(predict_step_k)
            
            print(f'step-{1+step}: ',_Format_Metrics(metrics_fn(predict_step_k,target_step_k)))
            