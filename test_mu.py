from model import *
import torch 
import gzip
import pickle
import os
import random
from helper import create_pyscipopt_with_x

ident = "lp"
idf = f"data_{ident}"

flist_test = os.listdir(f'./data_{ident}/test')

def restore_feas_MIS(A,x,y):
    min_vals = {}
    spa = A.to_sparse()
    idx = spa.indices()
    res = torch.zeros(x.shape)
    for i in range(idx.shape[1]//2):
        x1 = idx[1,2*i].item()
        x2 = idx[1,2*i+1].item()
        if x1 not in min_vals:
            min_vals[x1] = x[x1]
        if x2 not in min_vals:
            min_vals[x2] = x[x2]
        xsum = x[x1]+x[x2]
        if xsum>1.0:
            # print(x[x1].item(),x[x2].item(),xsum.item())
            newx1 = x[x1]/xsum
            newx2 = x[x2]/xsum
            min_vals[x1] = min(min_vals[x1], newx1)
            min_vals[x2] = min(min_vals[x2], newx2)
            new_sum = (min_vals[x1]+min_vals[x2])
            # print(min_vals[x1].item(),min_vals[x2].item(),new_sum.item())
            # print('-----------')
    for key in min_vals:
        res[key] = max(min_vals[key],0.0)
    return res

mdl = framework_fixed_mu(2,2,64,4)

if os.path.exists(f"./model/best_model_fixed_mu_{ident}.mdl"):
    checkpoint = torch.load(f"./model/best_model_fixed_mu_{ident}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')
else:
    quit()

loss_func = torch.nn.MSELoss()


flog = open(f'./logs/test_log_fixed_mu_{ident}.log','w')

eps=0.2
avg_loss=0
for indx, fnm in enumerate(flist_test):
    # test
    #  reading
    f = gzip.open(f'./data_{ident}/test/{fnm}','rb')
    A,v,c,sol,dual,obj = pickle.load(f)
    

    A = torch.as_tensor(A,dtype=torch.float32)

    amx = torch.max(A)
    m = A.shape[0]
    mu = 1/eps * torch.log(m*amx/eps)

    x = torch.as_tensor(v,dtype=torch.float32)
    y = torch.as_tensor(c,dtype=torch.float32)
    x_gt = torch.as_tensor(sol,dtype=torch.float32)
    y_gt = torch.as_tensor(dual,dtype=torch.float32)
    f.close()
    #  obtain loss
    x2,y = mdl(A,x,y,mu)
    x_res = x2
    if ident == 'mis':
        x_res = restore_feas_MIS(A,x2,y)
    for i in range(x_res.shape[0]):
        print(x_res[i],sol[i])

    loss = loss_func(x_res, x_gt)
    avg_loss += loss.item()

    _,_,obj2 = create_pyscipopt_with_x(A,x_res)

    print(f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}')
    # print(x)
    st = f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}\n'
    flog.write(st)
    flog.flush()



avg_loss /= round(len(flist_test),2)
print(f'Test Avg loss::::{avg_loss}')
st = f'Test Avg loss::::{avg_loss}\n'
flog.write(st)

flog.flush()


flog.close()