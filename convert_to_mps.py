import pyscipopt as scp
import torch
import pickle
import gzip
import random
import time
import os
import multiprocessing



def create_pyscipopt(A, mode='lp', filename = "gg", minA = 1.0):
    n = A.shape[1]
    m = A.shape[0]

    alist  = A.to_dense().tolist()
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
        obj += varis[j]*minA
    if mode == 'lp':
        model.setObjective(obj, sense = "maximize")
    else:
        model.setObjective(obj, sense = "minimize")


    path = f"/data3/lxyang/git/lq/packingalign/grb_folder/ins/{filename}.mps"

    model.writeProblem(path)


dirs = []
dirs.append(['/data3/lxyang/git/lq/packingalign/data_CA_20_20/test','lp','CA'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_CA_50_50/test','lp','CAmid'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_CA_400_400/test','lp','CAlg'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_lp_15_15/test','lp','lp'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_lp_75_75/test','lp','lpmid'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_lp_500_500/test','lp','lplg'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_IS_10/test','lp','IS'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_IS_50/test','lp','ISmid'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_IS_500/test','lp','ISlg'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_covering_15_15_60.0/test','covering','covering'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_covering_75_75_60.0/test','covering','coveringmid'])
dirs.append(['/data3/lxyang/git/lq/packingalign/data_covering_500_500_60.0/test','covering','coveringlg'])

for d in dirs:
    ident = d[0]
    files = os.listdir(ident)
    mode = d[1]
    pn = d[2]
    print(files)
    for idx,fnm in enumerate(files):

        f = gzip.open(f'{ident}/{fnm}','rb')
        # A,v,c,sol,dual,obj = pickle.load(f)
        tar = pickle.load(f)
        A = tar[0]
        A = torch.as_tensor(A,dtype=torch.float32)

        minA  = 1.0
        if len(tar)>=9:
            cost = torch.as_tensor(tar[7])
            minA = torch.as_tensor(tar[8])
        f.close()

        create_pyscipopt(A, mode=mode, filename = f'{pn}_{idx}', minA = minA)
