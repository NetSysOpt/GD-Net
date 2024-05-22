from model import *
import torch 
import gzip
import pickle
import os
import random
from helper import create_pyscipopt_with_x

ident = "lp"
# ident = "lp_20_20"
ident = "lp_75_75"
# ident = "lp_500_500"


ident = "lp_75_75"
model_name = 'lp_75_75'


idf = f"data_{ident}"


flist_test = os.listdir(f'./data_{ident}/test')



def restore_feas_LP(A,x,y=None):
    
    x=torch.clamp(x,min=0.0,max=1.0)
    
    min_vals = {}
    spa = A.to_sparse()
    idx = spa.indices()
    val = spa.values()
    res = torch.zeros(x.shape)
    
    row_id = 0
    buffer = []
    buffer_sum = 0.0
    
    ts2 = eval_feas(A,x)
    # print(max(ts2),min(ts2))
    
    
    for i in range(idx.shape[1]):
        current_idx = idx[1][i].item()
        if current_idx not in min_vals:
            min_vals[current_idx] = x[current_idx]
    
    for i in range(idx.shape[1]):
        current_row = idx[0][i].item()
        current_idx = idx[1][i].item()
        # print(current_row,current_idx)
        # print(f"    {A[current_row][current_idx]}  {val[i]}")
        # input()
        if row_id != current_row:
            if len(buffer)!=0 and buffer_sum>1.0:
                # new constraint, need to deal with buffer
                new_sum = 0.0
                # print(f"Row: {current_row}::: current sum:{round(buffer_sum.item(),2)}",end="   ")
                for b in buffer:
                    newx1 = min_vals[b]/(buffer_sum)
                    # print(f"      change: {b} from {min_vals[b].item()} to {newx1.item()}")
                    # input()
                    min_vals[b] = min(min_vals[b], newx1)
                    new_sum += min_vals[b] * A[row_id][b]
                # print('        new row val: ',new_sum.item())
            buffer = []
            buffer_sum = 0.0
            row_id = current_row
            
        buffer.append(current_idx)    
        buffer_sum += val[i]*min_vals[current_idx]    
        
    for key in min_vals:
        res[key] = max(min_vals[key],0.0)
        res[key] = min(min_vals[key],1.0)
    
    ts2 = eval_feas(A,res)
    # print(max(ts2),min(ts2))
    # quit()
    
    return res

def eval_feas(A,x):
    ad = A.to_dense()
    ts = torch.matmul(ad,x)
    return ts
    
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

mdl = GCN(1,1,64)

if os.path.exists(f"./model/best_GCN_{model_name}.mdl"):
    checkpoint = torch.load(f"./model/best_GCN_{model_name}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')
else:
    quit()

loss_func = torch.nn.MSELoss()


flog = open(f'./logs/test_log_fixed_GCN_{ident}.log','w')

eps=0.2
avg_loss=0



sum_obj = 0.0
sum_obj2 = 0.0
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
    n = A.shape[1]
    x = torch.zeros((m,1))
    # x = torch.ones((n,1))
    y = torch.zeros((m,1))
    
    x2 = mdl(A,x,y)
    x_res = x2
    
    # ts1 = eval_feas(A,x_res)
    if ident == 'mis':
        x_res = restore_feas_MIS(A,x2,y)
    elif 'lp' in ident:
        x_res = restore_feas_LP(A,x2,y)
        print('restored feasibility')
    # for i in range(x_res.shape[0]):
    #     print(x_res[i],sol[i])
    
    # # x_s = torch.div(x_res,ts2)
    # # ts3 = eval_feas(A,x_s)
    
    # for i in range(ts2.shape[0]):
    #     print(ts1[i],ts2[i])

    x_res = x_res.squeeze(-1)

    loss = loss_func(x_res, x_gt)
    avg_loss += loss.item()
    
    x_res = torch.clamp(x_res,min=0.0,max=1.0)

    
    # _,_,obj2 = create_pyscipopt_with_x(A,x_res)
    
    obj2 = min(torch.sum(x_res), obj)
    obj2 = torch.sum(x_res)

    print(f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}')
    # print(x)
    st = f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}\n'
    flog.write(st)
    flog.flush()
    
    sum_obj+=obj
    sum_obj2+=obj2



st = f'Last best val loss gen:  {best_loss}\n'
flog.write(st)

avg_loss /= round(len(flist_test),2)
print(f'Test Avg loss::::{avg_loss}')
st = f'Test Avg loss::::{avg_loss}\n'
flog.write(st)
sum_obj /= round(len(flist_test),2)
print(f'Avg Obj::::{sum_obj}')
st = f'Avg Obj::::{sum_obj}\n'
flog.write(st)

sum_obj2 /= round(len(flist_test),2)
print(f'Avg Predicted Obj::::{sum_obj2}')
st = f'Avg Predicted Obj::::{sum_obj2}\n'
flog.write(st)


flog.flush()


flog.close()