import pyscipopt as scp
import pickle
import gzip
import random
import time
import os
import torch
import multiprocessing
from ecole import RandomGenerator
from ecole.instance import SetCoverGenerator, CapacitatedFacilityLocationGenerator, IndependentSetGenerator, CombinatorialAuctionGenerator

rng = RandomGenerator(0)
cfg = CapacitatedFacilityLocationGenerator()
isg = IndependentSetGenerator()
cag = CombinatorialAuctionGenerator()


def ext(A):
    n = A.shape[1]
    m = A.shape[0]
    # avg coeff, coeff degree
    v_feat = torch.zeros(size=(n,2))
    # avg coeff, constraint degree
    c_feat = torch.zeros(size=(m,2))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j]>0:
                v_feat[j,0] += A[i,j]
                v_feat[j,1] += 1
                c_feat[i,0] += A[i,j]
                c_feat[i,1] += 1
        c_feat[i,0] /= c_feat[i,1]
    
    for j in range(A.shape[1]):
        v_feat[j,0] /= v_feat[j,1]
    
    return A,v_feat,c_feat



def normalize(A,b,c):
    n = A.shape[1]
    m = A.shape[0]

    new_values = []
    old_values = A.values()
    indc = A.indices()

    minA = 1e+20
    for idx in range(indc.shape[1]):
        i = indc[0,idx]
        j = indc[1,idx]
        bb = old_values[idx]/(b[i]*c[j])
        new_values.append(bb.item())
        if bb.item()>0 and bb.item() < minA:
            minA = bb.item()

    for idx in range(len(new_values)):
        new_values[idx] /= minA

    A = torch.sparse_coo_tensor(indc,new_values,size=[m,n])

    return A,minA


def getAmat(model):
    vs = model.getVars()
    vmap={}
    for v in vs:
        vmap[v.name] = v.getIndex()
    constr = model.getConss()
    indc = [[],[]]
    vals=[]
    m = len(constr)
    n = len(vs)
    for indx,c in enumerate(constr):
        coeff_dict = model.getValsLinear(c)
        for ele in coeff_dict:
            v = vmap[ele]
            indc[0].append(indx)
            indc[1].append(v)
            vals.append(coeff_dict[ele])
    A = torch.sparse_coo_tensor(indc,vals,size=[m,n])
    return A



def create_pyscipopt_Asparse(A,minA,mode='lp'):
    n = A.shape[1]
    m = A.shape[0]
    
    Aind = A.indices()
    Aval = A.values()
    nind = Aind.shape[1]

    varis = []
    cons = []
    model = scp.Model("CA")
    for i in range(n):
        varis.append(model.addVar(f"x_{i}"))

    if mode == 'lp':
        ridx = -1
        tmp_coeff = []
        tmp_var = []
        for idx in range(nind):
            if Aind[0,idx]!=ridx:
                # new constraint
                cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))<=1))
                ridx = Aind[0,idx]
                tmp_coeff = []
                tmp_var = []
            tmp_coeff.append(Aval[idx])
            tmp_var.append(varis[Aind[1,idx]])
        if len(tmp_coeff)!=0:
            cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))<=1))
            
    else:
        ridx = -1
        tmp_coeff = []
        tmp_var = []
        for idx in range(nind):
            if Aind[0,idx]!=ridx:
                # new constraint
                cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))>=1))
                ridx = Aind[0,idx]
                tmp_coeff = []
                tmp_var = []
            tmp_coeff.append(Aval[idx])
            tmp_var.append(varis[Aind[1,idx]])
        if len(tmp_coeff)!=0:
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
    objv = model.getObjVal()/minA
    return res,dual,objv,model.getTotalTime()



def create_pyscipopt(model):
    
    varis = model.getVars()
    for v in varis:
        model.chgVarType(v,'CONTINUOUS')
    model.optimize()
    cons = model.getConss()
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

def helper(ins,A):
    c = []
    b = []
    for v in ins.getVars():
        c.append(v.getObj())
    tt = A.to_dense().tolist()
    for cs in ins.getConss():
        b.append(ins.getRhs(cs))
    print(min(c),max(c))
    print(min(b),max(b))
    print(A.values().min(),A.values().max())
    quit()


def create_and_save_is(filename, n):

    ins = isg.generate_instance(n_nodes=n, affinity=2,rng =rng)
    ins = ins.as_pyscipopt()
    A = getAmat(ins)

    A = A.coalesce()
    A,v,c = ext(A)
    sol,dual,obj,time = create_pyscipopt(ins)
    to_pack = [A,v,c,sol,dual,obj,time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()
    print(f'Finished {filename}, time:{time}  obj:{obj}')


def create_pyscipopt_fromA(A, mode='covering'):
    n = A.shape[1]
    m = A.shape[0]

    alist  = A.to_dense().tolist()
    varis = []
    cons = []
    model = scp.Model("packing")
    for i in range(n):
        varis.append(model.addVar(f"x_{i}"))

    for i in range(m):
        # tmp_expr = scp.Expr()
        tmp_coeff = []
        tmp_var = []
        for j in range(n):
            if alist[i][j]>0:
                # tmp_expr += A[i,j]*varis[j]
                tmp_coeff.append(alist[i][j])
                tmp_var.append(varis[j])
        cons.append(model.addCons(scp.quicksum(tmp_coeff[i] * tmp_var[i] for i in range(len(tmp_coeff)))>=1))

    obj = scp.Expr()
    for j in range(n):
        obj += varis[j]
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

def create_and_save_LSD(filename, n):

    ins = isg.generate_instance(n_nodes=n, affinity=2,rng =rng)
    ins = ins.as_pyscipopt()
    A = getAmat(ins)

    A = A.transpose(0,1).coalesce()
    A,v,c = ext(A)
    sol,dual,obj,time = create_pyscipopt_fromA(A,'covering')
    to_pack = [A,v,c,sol,dual,obj,time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()
    print(f'Finished {filename}, time:{time}  obj:{obj}')


def generate_dateset_is(n=100,nmax=100,train_files=1000,valid_files=100,test_files=100):

    type_p = "IS"

    if not os.path.isdir(f"../data_{type_p}_{n}"):
        os.mkdir(f"../data_{type_p}_{n}")
        os.mkdir(f"../data_{type_p}_{n}/train")
        os.mkdir(f"../data_{type_p}_{n}/valid")
        os.mkdir(f"../data_{type_p}_{n}/test")



    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_is, args=(f"../data_{type_p}_{n}/train/prob_{i}.pkl",n1,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_is, args=(f"../data_{type_p}_{n}/valid/prob_{i}.pkl",n1,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_is, args=(f"../data_{type_p}_{n}/test/prob_{i}.pkl",n1,))


    pool.close()
    pool.join()



def generate_dateset_lsd(n=100,nmax=100,train_files=1000,valid_files=100,test_files=100):

    type_p = "LSD"

    if not os.path.isdir(f"../data_{type_p}_{n}"):
        os.mkdir(f"../data_{type_p}_{n}")
        os.mkdir(f"../data_{type_p}_{n}/train")
        os.mkdir(f"../data_{type_p}_{n}/valid")
        os.mkdir(f"../data_{type_p}_{n}/test")



    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_LSD, args=(f"../data_{type_p}_{n}/train/prob_{i}.pkl",n1,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_LSD, args=(f"../data_{type_p}_{n}/valid/prob_{i}.pkl",n1,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        pool.apply_async(create_and_save_LSD, args=(f"../data_{type_p}_{n}/test/prob_{i}.pkl",n1,))


    pool.close()
    pool.join()



def create_and_save_cfg(filename, n,m):

    ins = cfg.generate_instance(n_customers=n, n_facilities=m,rng =rng)
    ins = ins.as_pyscipopt()
    A = getAmat(ins)
    A = A.coalesce()
    A,v,c = ext(A)
    cost = []
    for v in ins.getVars():
        cost.append(v.getObj())
    sol,dual,obj,time = create_pyscipopt(ins)
    to_pack = [A,v,c,sol,dual,obj,time]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()



def generate_dateset_cfg(n=100,nmax=100,m=100,mmax=100,train_files=1000,valid_files=100,test_files=100):

    type_p = "CFG"

    if not os.path.isdir(f"../data_{type_p}_{n}_{m}"):
        os.mkdir(f"../data_{type_p}_{n}_{m}")
        os.mkdir(f"../data_{type_p}_{n}_{m}/train")
        os.mkdir(f"../data_{type_p}_{n}_{m}/valid")
        os.mkdir(f"../data_{type_p}_{n}_{m}/test")



    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_cfg, args=(f"../data_{type_p}_{n}_{m}/train/prob_{i}.pkl",n1,m1,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_cfg, args=(f"../data_{type_p}_{n}_{m}/valid/prob_{i}.pkl",n1,m1,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_cfg, args=(f"../data_{type_p}_{n}_{m}/test/prob_{i}.pkl",n1,m1,))


    pool.close()
    pool.join()







def create_and_save_ca(filename, n, m):
    ins = cag.generate_instance(n_items=n, n_bids=m,rng =rng)
    ins = ins.as_pyscipopt()
    cost = []
    for v in ins.getVars():
        cost.append(v.getObj())
    b = []
    for cc in ins.getConss():
        b.append(ins.getRhs(cc))

    A = getAmat(ins)
    A = A.coalesce()
    A,minA = normalize(A,b,cost)
    A = A.coalesce()


    A,v,c = ext(A)
    sol,dual,obj,time = create_pyscipopt_Asparse(A,minA)
    to_pack = [A,v,c,sol,dual,obj,time,cost,minA]
    f = gzip.open(filename,'wb')
    pickle.dump(to_pack,f)
    f.close()
    print(f'Finished {filename}, time:{time}  obj:{obj}')

# create_and_save_ca(f"../data_CA_20_20/train/prob_0.pkl",20,20)
# quit()

def generate_dateset_ca(n=100,nmax=100,m=100,mmax=100,train_files=1000,valid_files=100,test_files=100):

    type_p = "CA"

    if not os.path.isdir(f"../data_{type_p}_{n}_{m}"):
        os.mkdir(f"../data_{type_p}_{n}_{m}")
        os.mkdir(f"../data_{type_p}_{n}_{m}/train")
        os.mkdir(f"../data_{type_p}_{n}_{m}/valid")
        os.mkdir(f"../data_{type_p}_{n}_{m}/test")



    pool = multiprocessing.Pool(processes=50) 

    for i in range(train_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_ca, args=(f"../data_{type_p}_{n}_{m}/train/prob_{i}.pkl",n1,m1,))


    for i in range(valid_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_ca, args=(f"../data_{type_p}_{n}_{m}/valid/prob_{i}.pkl",n1,m1,))


    for i in range(test_files):
        n1 = random.randint(n, nmax)
        m1 = random.randint(m, mmax)
        pool.apply_async(create_and_save_ca, args=(f"../data_{type_p}_{n}_{m}/test/prob_{i}.pkl",n1,m1,))


    pool.close()
    pool.join()