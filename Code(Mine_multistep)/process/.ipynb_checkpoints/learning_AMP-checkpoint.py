import CITY
import PARAM as _PARAM
import PATH as _PATH
import torch as _torch
import os as _os
from time import time as _get_time
from process.loss import Format_Metrics as _Format_Metrics
from torch.utils.data import DataLoader as _DataLoader
from torchinfo import summary as _summary



def Train(model,optimizer,scaler,dataloader,loss_fn,metrics_fn,k):
    model.train()
    n=len(dataloader)
    metrics=[]
    t_start=_get_time()
    
    for (batch,data) in enumerate(dataloader):
        (input,target)=data
        with _torch.autocast(device_type=_PARAM.DEVICE,dtype=_torch.float16):
            predict=model([x.to(_PARAM.DEVICE) for x in input])
            target=target.permute(0,2,3,1).to(_PARAM.DEVICE)
            loss=loss_fn(predict,target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        metrics.append(metrics_fn(predict,target))
        
        if (1+batch)%k==0:
            t_end=_get_time()
            t=(t_end-t_start)/k
            t_start=t_end
            print(f'{1+batch:>5d}/{n} | {t:4.2f}s/B {t*k:2.0f}s/R {t*n/60:2.0f}m/E [{t*(n-batch-1)/60:2.0f}m] | '+\
                  _Format_Metrics(_torch.tensor(metrics[-k:]).mean(0).tolist()))

    return _torch.tensor(metrics).mean(0).tolist()


def Test(model,dataloader,metrics_fn):
    model.eval()
    n=len(dataloader)
    metrics=[]

    with _torch.inference_mode():
        for data in dataloader:
            (input,target)=data
            predict=model([x.to(_PARAM.DEVICE) for x in input])
            target=target.permute(0,2,3,1).to(_PARAM.DEVICE)
            metrics.append(metrics_fn(predict.where(predict>0,0),target))
    
    return _torch.tensor(metrics).mean(0).tolist()


def Loop(model,optimizer,scheduler,scaler,dataset,loss_fn,metrics_fn,
         batch_size=32,num_epochs=100,num_workers=0,k=100,load_checkpoint=False):
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
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            del checkpoint
            print(last_file+'\n')
    
    for epoch in range(start_epoch,1+num_epochs):
        print(f'Epoch: {epoch:0>3d}/{num_epochs:0>3d}    Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        metrics={'train':Train(model,optimizer,scaler,dataloader['train'],loss_fn,metrics_fn,k),
                 'validate':Test(model,dataloader['validate'],metrics_fn),
                 'test':Test(model,dataloader['test'],metrics_fn)}
        scheduler.step()
 
        _torch.save({'epoch':epoch,'metrics':metrics,
                     'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'scaler_state_dict':scaler.state_dict(),
                     'parameters':{'DIM_GLOBAL':_PARAM.DIM_GLOBAL,
                                   'DIM_EDGE':_PARAM.DIM_EDGE,
                                   'NUM_HEADs':_PARAM.NUM_HEADs,
                                   'NUM_BLOCKs':_PARAM.NUM_BLOCKs} if name=='Mine' else 0},
                    path+'/'+f'{name}_{int(_get_time())}_E{epoch:0>3d}_'+\
                    '_train_'+'_'.join([f'{x[0]:5.3f}' for x in metrics["train"]])+\
                    '_validate_'+'_'.join([f'{x[0]:5.3f}' for x in metrics["validate"]])+\
                    '_test_'+'_'.join([f'{x[0]:5.3f}' for x in metrics["test"]])+'.pt')

        print(f'   Train Loss:  {_Format_Metrics(metrics["train"])}\n'+\
              f'Validate Loss:  {_Format_Metrics(metrics["validate"])}\n'+\
              f'    Test Loss:  {_Format_Metrics(metrics["test"])}\n')
