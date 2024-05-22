from model import *
import torch 
import gzip
import pickle
import os
import random
import time
from helper import create_pyscipopt_with_x

ident = "lp_120_120"
ident = "lp_15_15"
ident = "lp"
# ident = "lp_20_20"
# ident = "lp_75_75"

model_name = "lp"
model_name = "lp_15_15"



model_name = "lp_75_75"
ident = "lp"

model_name = "lp_15_15"
ident = "lp_15_15"

# model_name = "lp_75_75"
# ident = "lp_75_75_50.0"

# model_name = "lp_75_75"
# ident = "lp_100_100_60.0"

# model_name = "lp_75_75"
# ident = "lp_75_75"


# model_name = "lp_500_500"
# ident = "lp_500_500"
exps = []

# exps.append(["lp_15_15","lp_15_15_60.0","15 test"])
# exps.append(["lp_75_75","lp_75_75_60.0","75 test"])
# exps.append(["lp_75_75","lp_75_75_50.0","75 gen to 50% test"])
# exps.append(["lp_75_75","lp_100_100_60.0","75 gen to 100 test"])
# exps.append(["lp_500_500","lp_500_500_60.0","500 test"])
eps=0.2
# eps=0.1
# exps.append(["CA_20_20","CA_20_20","CA test"])
# exps.append(["CA_50_50","CA_50_50","dchannel",5])
# exps.append(["CA_300_300","CA_300_300","CA test"])


# exps.append(["IS_10","IS_10","IS test"])
# exps.append(["IS_50","IS_50","IS test"])
# exps.append(["IS_500","IS_500","IS test"])


# exps.append(["CA_300_300","CA_400_400","CA test"])
# exps.append(["IS_500","IS_500","IS test"])
exps.append(["lp_1000_1000_60.0","lp_1000_1000_60.0","dchannel",5])
exps.append(["lp_500_500_60.0","lp_500_500_60.0","dchannel",5])
# exps.append(["lp_75_75_60.0","lp_75_75_60.0","dchannel",5])
# exps.append(["lp_500_500_60.0","lp_600_600_60.0","dchannel",5])
# exps.append(["IS_50","IS_50","dchannel",5])
# exps.append(["IS_500","IS_500","dchannel",5])
# exps.append(["IS_1000","IS_1000","dchannel",5])

# generalization
# exps.append(["lp_1000_1000_60.0","lp_1100_1100_60.0","dchannel",5])
# exps.append(["IS_1000","IS_1100","dchannel",5])
exps.append(["IS_1000","IS_1500","dchannel",5])
exps.append(["lp_1000_1000_60.0","lp_1500_1500_60.0","dchannel",5])


st_rec=[]
for ele in exps:
    # eps=0.01

    model_name = ele[0]
    ident = ele[1]
    model_type = ele[2]
    nfeat = ele[3]
    
    print(f'Current running:::: {ele[2]}')

    idf = f"data_{ident}"
    nfeat = 5
    other='x0'+model_type
    if nfeat!=1:
        other += f'_feat{nfeat}'
    if eps!=0.2:
        other += f'_ep{eps}'

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
                # new_sum = (min_vals[x1]+min_vals[x2])
                # print(min_vals[x1].item(),min_vals[x2].item(),new_sum.item())
                # print('-----------')
        for key in min_vals:
            res[key] = max(min_vals[key],0.0)
        return res


    def restore_feas_LP2(A,x,y=None,ub=1.0):
        
        if ub is None:
            x=torch.clamp(x,min=0.0)
        else:
            x=torch.clamp(x,min=0.0,max=ub)
        
        mma = torch.matmul(A,x).max()
        res = x/mma
        
        return res
    
    

    def restore_feas_LP(A,x,y=None,ub=1.0):
        
        if ub is None:
            x=torch.clamp(x,min=0.0)
        else:
            x=torch.clamp(x,min=0.0,max=ub)
        
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
            if ub is not None:
                res[key] = min(min_vals[key],ub)
        
        ts2 = eval_feas(A,res)
        # print(max(ts2),min(ts2))
        # quit()
        return res

    def eval_feas(A,x):
        ad = A.to_dense()
        ts = torch.matmul(ad,x)
        return ts
        
        
        
    # a1 = [[2.0,2.0],[1.0,1.0]]
    # x1 = [0.5,0.5]
    # a1 = torch.as_tensor(a1)
    # x1 = torch.as_tensor(x1)
    # xx = restore_feas_LP(a1,x1)
    # print(xx)
    # quit()


    # mdl = framework_model3(2,2,64,4)
    # mdl = framework_model1dim(4,64,nfeat)
    mdl = framework_model1dim(4,64,nfeat,model_type)

    print(f"./model/best_model3_{model_name}{other}.mdl")
    if os.path.exists(f"./model/best_model3_{model_name}{other}.mdl"):
        checkpoint = torch.load(f"./model/best_model3_{model_name}{other}.mdl")
        mdl.load_state_dict(checkpoint['model'])
        if 'nepoch' in checkpoint:
            last_epoch=checkpoint['nepoch']
        best_loss=checkpoint['best_loss']
        print(f'Last best val loss gen:  {best_loss}')
        print('Model Loaded')

        # for name,param in mdl.named_parameters():
        #     if param.requires_grad:
        #         print(name,param)
        # quit()
    else:
        print('Model not found')
        print(f"./model/best_model3_{model_name}{other}.mdl")
        quit()

    parm_num = count_parameters(mdl)

    print(f'Number of parameters:: {parm_num}')
    loss_func = torch.nn.MSELoss()

    file_name = f'{ident}'
    if ident!=model_name:
        file_name = file_name+f"_gen{model_name}"
    flog = open(f'./logs/test_log_model3_{file_name}{other}.log','w')
    flist_test.sort()
    avg_loss=0


    sum_obj = 0.0
    sum_obj2 = 0.0

    our_time = 0.0
    grb_time = 0.0

    avg_ratio = 0.0
    avg_gap = 0.0
    for indx, fnm in enumerate(flist_test):
        # test
        #  reading

        f = gzip.open(f'./data_{ident}/test/{fnm}','rb')
        # A,v,c,sol,dual,obj = pickle.load(f)
        tar = pickle.load(f)
        A = tar[0]
        v = tar[1]
        c = tar[2]
        sol = tar[3]
        dual = tar[4]
        obj = tar[5]
        sol_time = tar[6]
        

        if len(tar)>=9:
            cost = tar[7]
            minA = tar[8]

        A = torch.as_tensor(A,dtype=torch.float32)

        amx = None
        if A.is_sparse:
            amx = torch.max(A.values())
        else:
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
        x = torch.ones((n,1))
        if 'x0' in other:
            x = torch.zeros((n,1))
        y = torch.zeros((m,1))
        
        st = time.time()
        x2,y = mdl(A,x,y,mu)
        inf_time = time.time() - st
        
        x_res = x2
        # ts1 = eval_feas(A,x_res)
        st = time.time()
        if ident == 'mis':
            x_res = restore_feas_MIS(A,x2,y)
        elif 'lp' in ident:
            x_res = restore_feas_LP(A,x2,y)
            print('restored feasibility')
        elif 'CA' in ident:
            x_res = restore_feas_LP(A,x2,y)
            print('restored feasibility')
        elif 'IS' in ident:
            x_res = restore_feas_LP(A,x2,y)
            print('restored feasibility')
        feas_time = time.time() - st
        # for i in range(x_res.shape[0]):
        #     print(x_res[i],sol[i])
        
        # # x_s = torch.div(x_res,ts2)
        # # ts3 = eval_feas(A,x_s)
        
        # for i in range(ts2.shape[0]):
        #     print(ts1[i],ts2[i])

        x_res = x_res.squeeze(-1)
        # if len(tar)>=9:
        #     x_res = x_res/minA
        #     for i in range(x_res.shape[0]):
        #         x_res[i] = x_res[i]/cost[i]

        loss = loss_func(x_res, x_gt)
        avg_loss += loss.item()
        

        
        # _,_,obj2 = create_pyscipopt_with_x(A,x_res)
        
        obj2 = min(torch.sum(x_res), obj)
        obj2 = torch.sum(x_res)
        if len(tar)>=9:
            # obj2 = 0.0
            # for i in range(x_res.shape[0]):
            #     obj2 += (x_res[i]*cost[i]).item()
            obj2=obj2/minA


        our_time += inf_time+feas_time
        grb_time += sol_time
        avg_ratio += (obj-obj2)/obj
        avg_gap += obj-obj2
        print(f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{sol_time}')
        # print(x)
        st = f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{sol_time}\n'
        flog.write(st)

        flog.flush()
        
        sum_obj+=obj
        sum_obj2+=obj2
        
    st='------------------------------------------------------'
    st_rec.append(st)
    st = f'Last best val loss gen:  {best_loss}\n'
    flog.write(st)
    st_rec.append(st)


    st = f'Number of parameters:: {parm_num}\n'
    print(st)
    flog.write(st)
    st_rec.append(st)

    st = f'Last best val loss gen:  {best_loss}\n'
    print(st)
    flog.write(st)
    st_rec.append(st)

    avg_loss /= round(len(flist_test),2)
    print(f'Test Avg loss::::{avg_loss}')
    st = f'Test Avg loss::::{avg_loss}\n'
    flog.write(st)
    st_rec.append(st)

    sum_obj /= round(len(flist_test),2)
    print(f'Avg Obj::::{sum_obj}')
    st = f'Avg Obj::::{sum_obj}\n'
    flog.write(st)
    st_rec.append(st)

    sum_obj2 /= round(len(flist_test),2)
    print(f'Avg Predicted Obj::::{sum_obj2}')
    st = f'Avg Predicted Obj::::{sum_obj2}\n'
    flog.write(st)
    st_rec.append(st)



    avg_ratio /= round(len(flist_test),2)
    print(f'Avg ratio::::{avg_ratio}')
    st = f'Avg ratio::::{avg_ratio}\n'
    flog.write(st)
    st_rec.append(st)

    avg_gap /= round(len(flist_test),2)
    print(f'Avg gap::::{avg_gap}')
    st = f'Avg gap::::{avg_gap}\n'
    flog.write(st)
    st_rec.append(st)

    our_time /= len(flist_test)
    grb_time /= len(flist_test)
    print(f'Avg Our time::::{our_time}')
    st = f'Avg Our time::::{our_time}\n'
    flog.write(st)
    st_rec.append(st)
    print(f'Avg GRB time::::{grb_time}')
    st = f'Avg GRB time::::{grb_time}\n'
    flog.write(st)
    st_rec.append(st)

    flog.flush()


    flog.close()
    
for e in st_rec:
    print(e)