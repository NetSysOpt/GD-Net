import torch 
import gzip
import pickle
import os
import random
import time
import gurobipy as gp

mode = 'lp'
mode = 'covering'

# 0 psimplex
# 1 dsimplex
# 2 barrier
# def establish_grb(A,nthreads=8,method=-1,timelim=100.0):
def establish_grb(A,nthreads=1,method=0,timelim=100.0,indx=0):
    n = A.shape[1]
    m = A.shape[0]
    print(f'Building {indx}th model with {n} vars    {m} cons')
    model = gp.Model("lp1")
    model.Params.Threads = nthreads
    model.Params.Method = method
    # model.Params.TimeLimit = 100.0
    model.Params.TimeLimit = timelim
    vs = model.addVars(n, vtype=gp.GRB.CONTINUOUS)
    model.setObjective(vs.sum()*-1.0, gp.GRB.MINIMIZE)
    if A.is_sparse:
        A=A.to_dense()
    npy = A.numpy()
    model.addConstrs((gp.quicksum(vs[j] * npy[i,j] for j in range(n)) <= 1.0) for i in range(m))
    model.update()
    print('Finished building')
    model.write(f'./ins/test_{indx}.mps')
    
    
def establish_grb_covering(A,nthreads=1,method=0,timelim=100.0,indx=0):
    print('Building model')
    n = A.shape[1]
    m = A.shape[0]
    print(f'Building {indx}th model with {n} vars    {m} cons')
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
    model.write(f'./ins/test_{indx}.mps')

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
            if model.cbGet(gp.GRB.Callback.SPX_OBJVAL)>obj:
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
    
    
    
if mode == 'lp':
        
    exps = []
    eps=0.2
    # exps.append(["","lp_1000_1000_600.0","dchannel",5])
    exps.append(["","lp_5000_5000_20.0","dchannel",5])
    # exps.append(["","lp_10000_10000_5.0","dchannel",5])


    st_rec=[]
    for ele in exps:
        # eps=0.01

        model_name = ele[0]
        ident = ele[1]
        model_type = ele[2]
        nfeat = ele[3]
        
        print(f'Current running:::: {ele[2]}')
        flist_test = os.listdir(f'../data_{ident}/test')
        
        
        for indx, fnm in enumerate(flist_test):
            # test
            #  reading
            print(fnm)
            iidx = fnm.split('_')[-1].split('.')[0]
            iix = int(fnm.split('_')[-1].replace('.pkl',''))

            f = gzip.open(f'../data_{ident}/test/{fnm}','rb')
            tar = pickle.load(f)
            A = tar[0]
            f.close()
            
            establish_grb(A,indx=iidx)
else:    
    exps = []
    eps=0.2
    exps.append(["","covering_1000_1000_600.0","dchannel",5])
    # exps.append(["","covering_5000_5000_20.0","dchannel",5])
    # exps.append(["","covering_10000_10000_5.0","dchannel",5])


    st_rec=[]
    for ele in exps:
        # eps=0.01

        model_name = ele[0]
        ident = ele[1]
        model_type = ele[2]
        nfeat = ele[3]
        
        print(ident)
        print(f'COvering::::::  Current running:::: {ele[2]}')
        flist_test = os.listdir(f'../data_{ident}/test')
        
        
        for indx, fnm in enumerate(flist_test):
            # test
            #  reading
            iidx = fnm.split('_')[-1].split('.')[0]
            print(fnm,iidx)
            iix = int(fnm.split('_')[-1].replace('.pkl',''))
            f = gzip.open(f'../data_{ident}/test/{fnm}','rb')
            tar = pickle.load(f)
            A = tar[0]
            f.close()
            
            establish_grb_covering(A,indx=int(iidx))