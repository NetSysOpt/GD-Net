import pyscipopt as scp
import torch
import pickle
import gzip
import random
import time
import os
import multiprocessing
import numpy as np
import gurobipy as gp

# TODO::change to sparse

def ext(A):
    n = A.shape[1]
    m = A.shape[0]
    # avg coeff, coeff degree
    v_feat = torch.zeros(size=(n,2))
    # avg coeff, constraint degree
    c_feat = torch.zeros(size=(m,2))

    # Since we use all 0 start, this part is not needed
    # for i in range(m):
    #     for j in range(n):
    #         print(f'at ({i},{j})')
    #         if A[i,j]>0:
    #             v_feat[j,0] += A[i,j]
    #             v_feat[j,1] += 1
    #             c_feat[i,0] += A[i,j]
    #             c_feat[i,1] += 1
    #     c_feat[i,0] /= c_feat[i,1]
    
    # for j in range(A.shape[1]):
    #     v_feat[j,0] /= v_feat[j,1]
    
    return A,v_feat,c_feat





def getAmat(mode):
    constr = model.getConss()[0]
    coeff_dict = model.getValsLinear(constr)
    for ele in coeff_dict:
        print(ele)


def normalize(A,b,c):
    # normalize rows
    for i in range(A.shape[0]):
        A[i,:] /= b[i]
    # normalize cols
    for i in range(A.shape[1]):
        A[:,i] /= c[i]
    # divided by min aij
    min_val = torch.min(A)
    if min_val<1e-4:
        min_val = 1e-4

    A = A/min_val
    return A

def gen_packing_lp(n_items, n_bins, cost_mean=4, cost_std=2, item_mean=1, item_std=1,  bin_mean=2, bin_std=0.5):
    costs = torch.normal(cost_mean, cost_std, size=( n_items,1))
    costs = torch.clamp(costs, min=1e-3)
    caps = torch.normal(bin_mean,bin_std, size = (n_bins,1))
    caps = torch.clamp(caps, min=1e-3)
    occup = torch.normal(item_mean, item_std, size=( n_bins, n_items))
    occup = torch.clamp(occup, min=1e-3)
    occup[torch.where(occup<=1e-3)] = 0.0
    # print(torch.where(occup!=0.0)[0].shape)
    # quit()

    # return costs, occup, caps
    return normalize(occup, caps, costs)

def create_pyscipopt(A, mode='lp'):
    n = A.shape[1]
    m = A.shape[0]

    alist  = A.tolist()
    varis = []
    cons = []
    model = scp.Model("packing")
    for i in range(n):
        varis.append(model.addVar(f"x_{i}"))

    if mode == 'lp':
        for i in range(m):
            # tmp_expr = scp.Expr()
            tmp_coeff = []
            tmp_var = []
            for j in range(n):
                if alist[i][j]>0:
                    # tmp_expr += A[i,j]*varis[j]
                    tmp_coeff.append(alist[i][j])
                    tmp_var.append(varis[j])
            
            
            # cons.append(model.addCons(tmp_expr <= 1))
            cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))<=1))
    else:
        for i in range(m):
            # tmp_expr = scp.Expr()
            tmp_coeff = []
            tmp_var = []
            for j in range(n):
                if alist[i][j]>0:
                    # tmp_expr += A[i,j]*varis[j]
                    tmp_coeff.append(alist[i][j])
                    tmp_var.append(varis[j])
            
            
            # cons.append(model.addCons(tmp_expr <= 1))
            cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))>=1))

    obj = scp.Expr()
    for j in range(n):
        obj += varis[j]
    if mode == 'lp':
        model.setObjective(obj, sense = "maximize")
    else:
        model.setObjective(obj, sense = "minimize")

    print()
    model.optimize()
    # model.writeProblem('checker.lp')
    # quit()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(model.getVal(v))
    dual = []
    for c in cons:
        dual.append(model.getDualSolVal(c))
    objv = model.getObjVal()
    return res,dual,objv,model.getTotalTime()


def create_grb(A, mode='lp'):
    n = A.shape[1]
    m = A.shape[0]

    alist  = A.tolist()
    varis = []
    cons = []
    model = gp.Model("packing")
    model.Params.Threads = 2
    for i in range(n):
        varis.append(model.addVar(vtype=gp.GRB.CONTINUOUS,name=f"x_{i}"))

    if mode == 'lp':
        for i in range(m):
            # tmp_expr = scp.Expr()
            tmp_coeff = []
            tmp_var = []
            for j in range(n):
                if alist[i][j]>0:
                    # tmp_expr += A[i,j]*varis[j]
                    tmp_coeff.append(alist[i][j])
                    tmp_var.append(varis[j])
            
            
            # cons.append(model.addCons(tmp_expr <= 1))
            cons.append(model.addConstr(gp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))<=1.0))
    else:
        for i in range(m):
            # tmp_expr = scp.Expr()
            tmp_coeff = []
            tmp_var = []
            for j in range(n):
                if alist[i][j]>0:
                    # tmp_expr += A[i,j]*varis[j]
                    tmp_coeff.append(alist[i][j])
                    tmp_var.append(varis[j])
            
            
            # cons.append(model.addCons(tmp_expr <= 1))
            cons.append(model.addConstr(gp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))>=1.0))

    obj = gp.LinExpr()
    for j in range(n):
        obj += varis[j]
    if mode == 'lp':
        model.setObjective(obj, gp.GRB.MAXIMIZE)
    else:
        model.setObjective(obj, gp.GRB.MINIMIZE)

    print()
    model.optimize()
    # model.writeProblem('checker.lp')
    # quit()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(v.X)
    dual = []
    for c in cons:
        dual.append(c.Pi)
    objv = model.ObjVal
    return res,dual,objv,model.Runtime

def create_pyscipopt_with_x(A,x):
    n = A.shape[1]
    m = A.shape[0]
    varis = []
    cons = []


    model = scp.Model("packing")
    for i in range(n):
        varis.append(model.addVar(f"x_{i}"))
    
    for i in range(m):
        tmp_expr = scp.Expr()
        for j in range(n):
            if A[i,j]>0:
                tmp_expr += A[i,j]*varis[j]

        cons.append(model.addCons(tmp_expr <= 1))

    obj = scp.Expr()
    for j in range(n):
        obj += varis[j]
    model.setObjective(obj, sense = "maximize")
    # fix by x
    for indx, v in enumerate(varis):
        model.fixVar(v,x[indx])

    print()


    model.optimize()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(model.getVal(v))
    dual = []
    for c in cons:
        dual.append(model.getDualSolVal(c))
    objv = model.getObjVal()
    return res,dual,objv


            
def create_and_save_binpacking(filename, n,m):
    A = gen_packing_lp(n,m)
    A,v,c = ext(A)
    sol,dual,obj,sol_time = create_pyscipopt(A)
    to_pack = [A,v,c,sol,dual,obj,sol_time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()

def generate_dateset_binpacking(n=100,m=80,train_files=1000,valid_files=100,test_files=100):
    for i in range(train_files):
        create_and_save_binpacking(f"./data/train/prob_{i}.pkl",n,m)
    for i in range(valid_files):
        create_and_save_binpacking(f"./data/valid/prob_{i}.pkl",n,m)
    for i in range(test_files):
        create_and_save_binpacking(f"./data/test/prob_{i}.pkl",n,m)






random.seed(0)

def gen_mis_lp(n,density):
    density += random.random()*0.3
    density -= random.random()*0.3
    elements = []
    for i in range(n):
        counter = 0
        for j in range(i+1,n):
            if random.random()<=density:
                elements.append([i,j])
                counter+=1
        if counter==0:
            elements.append([i,int(round(random.random()*(n-1),0))])
    A = torch.zeros(size=(len(elements),n))
    for indx,ele in enumerate(elements):
        A[indx,ele[0]] = 1.0
        A[indx,ele[1]] = 1.0
                
    return A
            

            
def create_and_save_MIS(filename, n,density):
    A = gen_mis_lp(n,density)
    A,v,c = ext(A)
    sol,dual,obj,time = create_pyscipopt(A)
    to_pack = [A,v,c,sol,dual,obj,sol_time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()

def generate_dateset_MIS(n=100,density=0.9,train_files=1000,valid_files=100,test_files=100):
    for i in range(train_files):
        create_and_save_MIS(f"./data_mis/train/prob_{i}.pkl",n,density)
    for i in range(valid_files):
        create_and_save_MIS(f"./data_mis/valid/prob_{i}.pkl",n,density)
    for i in range(test_files):
        create_and_save_MIS(f"./data_mis/test/prob_{i}.pkl",n,density)


            

def gen_gen_lp(n,m,density):
    A = np.zeros((m,n))
    density_map = np.random.rand(m,n)
    val_map = np.random.rand(m,n)
    A = np.where(density_map <= density, val_map, 0.0)
    # for i in range(m):
    #     for j in range(n):
    #         if random.random()<=density:
    #             # gen num
    #             A[i,j] = random.random()+1.0
    #         # else:
    #         #     A[i,j] = 0.0
    A = torch.as_tensor(A)
    return A

def create_and_save_lp(filename, n,m,density):
    A = gen_gen_lp(n,m,density)
    print(f'Generating {filename}')
    A,v,c = ext(A)
    # sol,dual,obj,sol_time = create_pyscipopt(A)
    sol,dual,obj,sol_time = create_grb(A)
    to_pack = [A,v,c,sol,dual,obj,sol_time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()
    print(f'Finished creating {filename}')

def generate_dateset_LP(n=100,nmax=100,m=100,mmax=100,p=0.9,train_files=1000,valid_files=100,test_files=100):
    pillar = f'{round(p*1000,0)}'
    if not os.path.isdir(f"./data_lp_{n}_{m}_{pillar}"):
        os.mkdir(f"./data_lp_{n}_{m}_{pillar}")
        os.mkdir(f"./data_lp_{n}_{m}_{pillar}/train")
        os.mkdir(f"./data_lp_{n}_{m}_{pillar}/valid")
        os.mkdir(f"./data_lp_{n}_{m}_{pillar}/test")

    # n1 = random.randint(n, nmax)
    # m1 = random.randint(m, mmax)
    # create_and_save_lp(f"./data_lp_{n}_{m}_{pillar}/train/prob_0.pkl",n1,m1,p,)
    # quit()

    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        # create_and_save_lp(f"./data_lp_{n}_{m}/train/prob_{i}.pkl",n1,m1,p)

        pool.apply_async(create_and_save_lp, args=(f"./data_lp_{n}_{m}_{pillar}/train/prob_{i}.pkl",n1,m1,p,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        # create_and_save_lp(f"./data_lp_{n}_{m}/valid/prob_{i}.pkl",n1,m1,p)
        pool.apply_async(create_and_save_lp, args=(f"./data_lp_{n}_{m}_{pillar}/valid/prob_{i}.pkl",n1,m1,p,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        # create_and_save_lp(f"./data_lp_{n}_{m}/test/prob_{i}.pkl",n1,m1,p)
        pool.apply_async(create_and_save_lp, args=(f"./data_lp_{n}_{m}_{pillar}/test/prob_{i}.pkl",n1,m1,p,))


    pool.close()
    pool.join()



def create_and_save_covering(filename, n,m,density):
    A = gen_gen_lp(n,m,density)
    A,v,c = ext(A)
    # sol,dual,obj,time = create_pyscipopt(A,mode='covering')
    sol,dual,obj,time = create_grb(A,mode='covering')
    to_pack = [A,v,c,sol,dual,obj,time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()

def generate_dateset_Covering(n=100,nmax=100,m=100,mmax=100,p=0.9,train_files=1000,valid_files=100,test_files=100):
    pillar = f'{round(p*1000,0)}'
    if not os.path.isdir(f"./data_covering_{n}_{m}_{pillar}"):
        os.mkdir(f"./data_covering_{n}_{m}_{pillar}")
        os.mkdir(f"./data_covering_{n}_{m}_{pillar}/train")
        os.mkdir(f"./data_covering_{n}_{m}_{pillar}/valid")
        os.mkdir(f"./data_covering_{n}_{m}_{pillar}/test")

    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_covering, args=(f"./data_covering_{n}_{m}_{pillar}/train/prob_{i}.pkl",n1,m1,p,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_covering, args=(f"./data_covering_{n}_{m}_{pillar}/valid/prob_{i}.pkl",n1,m1,p,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_covering, args=(f"./data_covering_{n}_{m}_{pillar}/test/prob_{i}.pkl",n1,m1,p,))


    pool.close()
    pool.join()





def create_grb_mf(n,m,p):
    # generate graph
    adj = []
    for i in range(n):
        for j in range(m):
            if random.random()<=p:
                adj.append((i,j))

    # randomize c_max

    nnz = len(adj)
    # A = torch.zeros(size=(n+m,nnz))
    Aindx = [[],[]]
    Aval = []

    varis = []
    cons = []
    model = gp.Model("maxflow")
    model.Params.Method = 0

    model.Params.Threads = 2

    tar_map_left = {}
    tar_map_right = {}

    nedge = 0
    edg_pos = {} 
    for edge in adj:
        c_max = random.random()*1.0+1.0
        # c_max = 1.0
        i = edge[0]
        j = edge[1]
        x = model.addVar(vtype=gp.GRB.CONTINUOUS,name=f"x_{i}_{j}",lb=0.0,ub=c_max)
        # print(f'x_{i}_{j}')
        edg_pos[f'x_{i}_{j}'] = nedge
        varis.append(x)
        if i not in tar_map_left:
            tar_map_left[i] = []
        if j not in tar_map_right:
            tar_map_right[j] = []
        tar_map_left[i].append(x)
        tar_map_right[j].append(x)
        nedge+=1
    multiplier = 40.0
    offset = 40
    model.update()

    n_cons = 0
    # left nodes 
    for i in range(n):
        if  i not in tar_map_left:
            continue
        involve_edges = tar_map_left[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        # cap = 2.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints for left nodes')
    
    # right nodes 
    for i in range(m):
        if  i not in tar_map_right:
            continue
        involve_edges = tar_map_right[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        # cap = 2.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints in total')
        
    obj = gp.LinExpr()
    for v in varis:
        obj += v
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    print()
    model.optimize()
    # model.writeProblem('checker.lp')
    # quit()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(v.X)
    dual = []
    for c in cons:
        dual.append(c.Pi)
    objv = model.ObjVal
    A = torch.sparse_coo_tensor(Aindx, Aval, [n_cons, nnz])
    return res,dual,objv,model.Runtime,A




def create_grb_mf2(n,m,p):
    # generate graph
    ngrp = 4

    left_sample = [x for x in range(n)]
    right_sample = [x for x in range(m)]
    
    left_samples=[]
    right_samples=[]

    left_size = n//ngrp
    rigth_size = m//ngrp

    random.shuffle(left_samples)
    random.shuffle(right_samples)

    for igrp in range(ngrp):
        left_samples.append(set(left_sample[igrp*left_size:igrp*left_size+left_size]))
        right_samples.append(set(right_sample[igrp*rigth_size:igrp*rigth_size+rigth_size]))

    adj = []
    for i in range(n):
        for j in range(m):
            lgrp = None
            rgrp = None
            for igrp in range(ngrp):
                if i in left_samples[igrp]:
                    lgrp = igrp
                if j in right_samples[igrp]:
                    rgrp = igrp
            if lgrp is None or rgrp is None:
                continue
            if (lgrp!=rgrp):
                adj.append([i,j])

    # randomize c_max

    nnz = len(adj)
    # A = torch.zeros(size=(n+m,nnz))
    Aindx = [[],[]]
    Aval = []

    varis = []
    cons = []
    model = gp.Model("maxflow")
    model.Params.Method = 0

    model.Params.Threads = 2

    tar_map_left = {}
    tar_map_right = {}

    nedge = 0
    edg_pos = {} 
    for edge in adj:
        c_max = random.random()*3.0+4.0
        # c_max = 1.0
        i = edge[0]
        j = edge[1]
        x = model.addVar(vtype=gp.GRB.CONTINUOUS,name=f"x_{i}_{j}",lb=0.0,ub=c_max)
        # print(f'x_{i}_{j}')
        edg_pos[f'x_{i}_{j}'] = nedge
        varis.append(x)
        if i not in tar_map_left:
            tar_map_left[i] = []
        if j not in tar_map_right:
            tar_map_right[j] = []
        tar_map_left[i].append(x)
        tar_map_right[j].append(x)
        nedge+=1
    multiplier = 240.0
    offset = 240
    model.update()

    n_cons = 0
    # left nodes 
    for i in range(n):
        if  i not in tar_map_left:
            continue
        involve_edges = tar_map_left[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        # cap = 1.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints for left nodes')
    
    # right nodes 
    for i in range(m):
        if  i not in tar_map_right:
            continue
        involve_edges = tar_map_right[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        # cap = 1.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints in total')
        
    obj = gp.LinExpr()
    for v in varis:
        obj += v
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    print()
    model.optimize()
    # model.writeProblem('checker.lp')
    # quit()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(v.X)
    dual = []
    for c in cons:
        dual.append(c.Pi)
    objv = model.ObjVal
    A = torch.sparse_coo_tensor(Aindx, Aval, [n_cons, nnz])
    return res,dual,objv,model.Runtime,A




def create_grb_mf3(n,m,p):
    # generate graph

    adj = []
    for i in range(n):
        adj.append([i,i])
        adj.append([i,i+n//2])
        adj.append([i+n//2,i])

    # randomize c_max

    nnz = len(adj)
    # A = torch.zeros(size=(n+m,nnz))
    Aindx = [[],[]]
    Aval = []

    varis = []
    cons = []
    model = gp.Model("maxflow")
    model.Params.Method = 0

    model.Params.Threads = 2

    tar_map_left = {}
    tar_map_right = {}

    nedge = 0
    edg_pos = {} 
    for edge in adj:
        c_max = random.random()*3.6+0.0
        c_max = 1.0
        i = edge[0]
        j = edge[1]
        x = model.addVar(vtype=gp.GRB.CONTINUOUS,name=f"x_{i}_{j}",lb=0.0,ub=c_max)
        # print(f'x_{i}_{j}')
        edg_pos[f'x_{i}_{j}'] = nedge
        varis.append(x)
        if i not in tar_map_left:
            tar_map_left[i] = []
        if j not in tar_map_right:
            tar_map_right[j] = []
        tar_map_left[i].append(x)
        tar_map_right[j].append(x)
        nedge+=1
    multiplier = 480.0
    offset = 20
    model.update()

    n_cons = 0
    # left nodes 
    orders = [x for x in range(n)]
    random.shuffle(orders)
    for i in orders:
        if  i not in tar_map_left:
            continue
        involve_edges = tar_map_left[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        cap = 1.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints for left nodes')
    
    # right nodes 
    orders = [x for x in range(m)]
    random.shuffle(orders)
    for i in range(m):
        if  i not in tar_map_right:
            continue
        involve_edges = tar_map_right[i]
        if len(involve_edges)==0:
            continue
        cap = random.random()*multiplier+offset
        cap = 1.0 
        # print(f'right node {i} edges: {involve_edges}   cap:{cap}')
        cons.append(model.addConstr(gp.quicksum(involve_edges)<=cap))
        for vv in involve_edges:
            pos = edg_pos[vv.VarName]
            Aindx[0].append(n_cons)
            Aindx[1].append(pos)
            Aval.append(1.0/cap)
        n_cons+=1
    print(f'Processed {n_cons} constraints in total')
        
    obj = gp.LinExpr()
    for v in varis:
        obj += v
    model.setObjective(obj, gp.GRB.MAXIMIZE)

    print()
    model.optimize()
    # model.writeProblem('checker.lp')
    # quit()
    print('Solve Finished')
    res = []
    for v in varis:
        res.append(v.X)
    dual = []
    for c in cons:
        dual.append(c.Pi)
    objv = model.ObjVal
    A = torch.sparse_coo_tensor(Aindx, Aval, [n_cons, nnz])
    return res,dual,objv,model.Runtime,A

def create_and_save_mf(filename, n,m,density,mode=0):
    print(f'Generating {filename}')
    # need to generate 
    # A,v,c = ext(A)
    # sol,dual,obj,sol_time = create_pyscipopt(A)
    if mode==0:
        sol,dual,obj,sol_time,A = create_grb_mf(n,m,density)
    elif mode ==1:
        sol,dual,obj,sol_time,A = create_grb_mf2(n,m,density)
    elif mode ==2:
        sol,dual,obj,sol_time,A = create_grb_mf3(n,m,density)
    A = A.coalesce()
    v = torch.zeros([A.shape[1],2])
    c = torch.zeros([A.shape[0],2])
    to_pack = [A,v,c,sol,dual,obj,sol_time]
    if filename!='':
        f = gzip.open(filename,'wb')
        pickle.dump(to_pack,f)
        f.close()
    print(f'Finished creating {filename}')

# create_and_save_mf('',2000,2000,0.6,mode=0)
# create_and_save_mf('',2000,2000,0.6,mode=1)
# create_and_save_mf('',600000,600000,0.6,mode=0)
# quit()


def generate_dateset_Maxflow(n=200,m=200,p=0.1,train_files=1000,valid_files=100,test_files=100,mode=0):
    pillar = f'{round(p*1000,0)}'
    if not os.path.isdir(f"./data_maxflow_{n}_{m}_{pillar}"):
        os.mkdir(f"./data_maxflow_{n}_{m}_{pillar}")
        os.mkdir(f"./data_maxflow_{n}_{m}_{pillar}/train")
        os.mkdir(f"./data_maxflow_{n}_{m}_{pillar}/valid")
        os.mkdir(f"./data_maxflow_{n}_{m}_{pillar}/test")

    
        



    pool = multiprocessing.Pool(processes=4) 

    for i in range(train_files):
        # create_and_save_lp(f"./data_lp_{n}_{m}/train/prob_{i}.pkl",n1,m1,p)

        pool.apply_async(create_and_save_mf, args=(f"./data_maxflow_{n}_{m}_{pillar}/train/prob_{i}.pkl",n,m,p,mode,))


    for i in range(valid_files):
        # create_and_save_lp(f"./data_lp_{n}_{m}/valid/prob_{i}.pkl",n1,m1,p)
        pool.apply_async(create_and_save_mf, args=(f"./data_maxflow_{n}_{m}_{pillar}/valid/prob_{i}.pkl",n,m,p,mode,))


    for i in range(test_files):
        # create_and_save_lp(f"./data_lp_{n}_{m}/test/prob_{i}.pkl",n1,m1,p)
        pool.apply_async(create_and_save_mf, args=(f"./data_maxflow_{n}_{m}_{pillar}/test/prob_{i}.pkl",n,m,p,mode,))


    pool.close()
    pool.join()