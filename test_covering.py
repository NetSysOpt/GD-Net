from model import *
import torch 
import gzip
import pickle
import os
import random
from helper import create_pyscipopt_with_x
import time
import gurobipy as gp
exps = []

# exps.append(["covering_15_15_60.0","covering_15_15_60.0","15 60 test"])
# exps.append(["covering_75_75_60.0","covering_75_75_60.0","dchannel",5])
# exps.append(["covering_500_500_60.0","covering_500_500_60.0","dchannel",5])
# exps.append(["covering_75_75_60.0","covering_75_75_60.0","",1])
# exps.append(["covering_1000_1000_60.0","covering_1000_1000_60.0","dchannel",5])
# exps.append(["covering_1000_1000_60.0","covering_1000_1000_60.0","",1])
# exps.append(["covering_75_75_50.0","covering_75_75_60.0","75 60 to 75 50 test"])
# exps.append(["covering_500_500_60.0","covering_500_500_60.0","75 60 to 100 60 test"])
# exps.append(["covering_600_600_60.0","covering_500_500_60.0","500 60 test"])
# exps.append(["LSD_50","LSD_50","dchannel",5])
# exps.append(["LSD_500","LSD_500","dchannel",5])
# exps.append(["LSD_1000","LSD_1000","dchannel",5])
# exps.append(["covering_75_75_60.0","covering_75_75_60.0","dchannel",5])

# generalization
# exps.append(["LSD_1100","LSD_1000","dchannel",5])
# exps.append(["covering_1100_1100_60.0","covering_1000_1000_60.0","dchannel",5])
# exps.append(["LSD_1500","LSD_1000","dchannel",5])
# exps.append(["covering_1500_1500_60.0","covering_1000_1000_60.0","dchannel",5])
exps.append(["covering_10000_10000_5.0","covering_10000_10000_5.0","dchannel",5])


# 0 psimplex
# 1 dsimplex
# 2 barrier
# def establish_grb(A,nthreads=8,method=-1,timelim=100.0):
def establish_grb(A,nthreads=1,method=0,timelim=100.0):
    print('Building model')
    n = A.shape[1]
    m = A.shape[0]
    model = gp.Model("lp1")
    model.Params.Threads = nthreads
    model.Params.Method = method
    # model.Params.TimeLimit = 100.0
    model.Params.TimeLimit = timelim
    vs = model.addVars(n, vtype=gp.GRB.CONTINUOUS)
    model.setObjective(vs.sum(), gp.GRB.MINIMIZE)
    npy = A.numpy()
    model.addConstrs((gp.quicksum(vs[j] * npy[i,j] for j in range(n)) >= 1.0) for i in range(m))
    model.update()
    print('Finished building')
    return model

def solve_ws(A,x,y,model):
    model.reset()
    model.Params.Method = 0
    model.Params.Presolve = 1
    model.Params.LPWarmStart = 2
    model.Params.LogToConsole = 1
    model.Params.TimeLimit = 100.0
    vs = model.getVars()
    for i,v in enumerate(vs):
        # print(f'{v} {x[i].item()}')
        v.PStart = x[i].item()
    model.optimize()
    new_obj = model.ObjVal
    new_time = model.Runtime
    print(f'ws obj: {new_obj}')
    print(f'ws time: {new_time}')
    return new_obj,new_time    
    
def solve_grb(A,x,y,model):
    model.reset()
    vs = model.getVars()
    model.optimize()
    ori_obj = model.ObjVal
    ori_time = model.Runtime
    print(f'ori obj: {ori_obj}')
    print(f'ori time: {ori_time}')
    return ori_obj, ori_time

def solve_grb_bestobj(A,x,y,model,obj):
    def cbk(model,where):
        if where == gp.GRB.Callback.SIMPLEX:
            if model.cbGet(gp.GRB.Callback.SPX_OBJVAL)<obj:
                model.terminate()
    model.reset()
    model.optimize(cbk)
    ori_obj = model.ObjVal
    ori_time = model.Runtime
    print(f'ori bestobj: {ori_obj}')
    print(f'ori bestobj time: {ori_time}')
    return ori_obj, ori_time
    

def solve_search(A,x,y,model,eps=20.0):
    model.reset()
    aux_vars = []
    for i,v in enumerate(vs):
        vs[v].PStart = x[i]
        tmp_var = model.addVar(vtype=gp.GRB.CONTINUOUS)
        aux_vars.append(tmp_var)
        model.addConstr(vs[v]-x[i]<=tmp_var)
        model.addConstr(x[i]-vs[v]<=tmp_var)
    model.addConstr(gp.quicksum(aux_vars)<=eps)
    
    print('Finished building')
    model.optimize()
    new_obj = model.ObjVal
    new_time = model.Runtime
    # print(f'ori obj: {ori_obj}')
    # print(f'ori time: {ori_time}')
    print(f'search obj: {new_obj}')
    print(f'search time: {new_time}')
    quit()
    



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
            # print(f"Row: {current_row}::: current sum:{round(buffer_sum.item(),2)}",end="   \n")
            # input()
            if len(buffer)!=0 and buffer_sum<1.0:
                # new constraint, need to deal with buffer
                for b in buffer:
                    newx1 = min_vals[b]/(buffer_sum)
                    # print(f"      change: {b} from {min_vals[b].item()} to {newx1.item()}")
                    # input()
                    min_vals[b] = max(min_vals[b], newx1)
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
    # input()
    # quit()
    
    return res

def eval_feas(A,x):
    ad = A.to_dense()
    ts = torch.matmul(ad,x)
    return ts
    
st_rec = []
for ele in exps:

    ident = ele[0]
    model_name = ele[1]
    model_type = ele[2]
    nfeat = ele[3]
    print(f'Current running:::: {ele[2]}')

    

    
    
    idf = f"data_{ident}"
    
    
    
    other=f'{model_type}'
    if nfeat!=1:
        other += f'_feat{nfeat}'

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


        
        
        
    # a1 = [[2.0,2.0],[1.0,1.0]]
    # x1 = [0.5,0.5]
    # a1 = torch.as_tensor(a1)
    # x1 = torch.as_tensor(x1)
    # xx = restore_feas_LP(a1,x1)
    # print(xx)
    # quit()


    # mdl = framework_model3(2,2,64,4)
    mdl = framework_model1dim_covering(4,64,nfeat)
    if 'dchannel' == model_type:
        print('!!!!!!!!!!USING Dchannel model, method 2')
        mdl = framework_model1dim_covering(4,64,nfeat,mode=model_type)

    print(f"./model/best_covering_{model_name}{other}.mdl")
    if os.path.exists(f"./model/best_covering_{model_name}{other}.mdl"):
        checkpoint = torch.load(f"./model/best_covering_{model_name}{other}.mdl")
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
        print('model not found')
        quit()

    loss_func = torch.nn.MSELoss()

    file_name = f'{ident}'
    if ident!=model_name:
        file_name = file_name+f"_gen{model_name}"
    flog = open(f'./logs/test_log_covering_{file_name}{other}.log','w')
    flist_test.sort()
    eps=0.2
    avg_loss=0


    sum_obj = 0.0
    sum_obj2 = 0.0

    our_time = 0.0
    grb_time = 0.0

    avg_ratio = 0.0
    avg_gap = 0.0

    avg_grb_sametle = 0.0
    avg_grb_beststop_time = 0.0

    grbtimelim = 200.0
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

        ori_time = tar[6]
        

        A = torch.as_tensor(A,dtype=torch.float32)

        if A.is_sparse:
            amx = torch.max(A.to_dense())
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
        st = time.time()
        # ts1 = eval_feas(A,x_res)
        if ident == 'mis':
            x_res = restore_feas_MIS(A,x2,y)
        elif 'covering' in ident or 'LSD' in ident:
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

        loss = loss_func(x_res, x_gt)
        avg_loss += loss.item()
        
        x_res = torch.clamp(x_res,min=0.0,max=1.0)

        
        # _,_,obj2 = create_pyscipopt_with_x(A,x_res)
        
        obj2 = min(torch.sum(x_res), obj)
        obj2 = torch.sum(x_res)

        our_time += inf_time+feas_time
        grb_time += ori_time

        avg_ratio += (obj2-obj)/obj
        avg_gap += obj2-obj
        
        model = establish_grb(A,timelim=inf_time+feas_time)
        grb_obj, grb_time = solve_grb(A,x,y,model)
        model.Params.TimeLimit = grbtimelim
        grb_100_obj, grb_100_time = solve_grb_bestobj(A,x,y,model,obj2)
        print(grb_obj, grb_time)

        # ws_obj, ws_time = solve_ws(A,x,y,model)
        # print(f'Warmstart: {ws_obj},{ws_time}')
        # quit()

        avg_grb_sametle += grb_obj
        avg_grb_beststop_time += grb_100_time
        
        
        print(f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{ori_time}\n  :::GRB:{grb_obj},{grb_100_time}')
        # print(x)
        st = f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   :::TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{ori_time} :::GRB:{grb_obj},{grb_100_time}\n'
        flog.write(st)
        flog.flush()
        
        sum_obj+=obj
        sum_obj2+=obj2
        



    st_rec.append(f'----------------{ident}----------------')
    st = f'Last best val loss gen:  {best_loss}\n'
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

    sum_obj2 /= round(len(flist_test),2)
    print(f'Avg Predicted Obj::::{sum_obj2}')
    st = f'Avg Predicted Obj::::{sum_obj2}\n'
    flog.write(st)
    st_rec.append(st)


    our_time /= len(flist_test)
    grb_time /= len(flist_test)
    print(f'Avg Our time::::{our_time}')
    st = f'Avg Our time::::{our_time}\n'
    flog.write(st)
    print(f'Avg GRB time::::{grb_time}')
    st = f'Avg GRB time::::{grb_time}\n'
    flog.write(st)
    st_rec.append(st)



    avg_grb_sametle /= len(flist_test)
    print(f'GRB at same time::::{avg_grb_sametle}')
    st = f'GRB at same time::::{avg_grb_sametle}\n'
    flog.write(st)
    avg_grb_beststop_time /= len(flist_test)
    print(f'GRB with same obj time::::{avg_grb_beststop_time}')
    st = f'GRB with same obj time::::{avg_grb_beststop_time}\n'
    flog.write(st)

    flog.flush()


    flog.close()
    
for e in st_rec:
    print(e)