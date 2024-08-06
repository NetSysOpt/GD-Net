import argparse
import copy
import time
import gzip
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from models import DeeperGCN
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from helper import *
import os
import shutil
from alive_progress import alive_bar
import gurobipy as gp
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def restore_feas_LP(A,x,y=None,ub=1.0):
    # print(x)
    # quit()
    st = time.time()
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
    
    # print(max(ts2),min(ts2))
    
    nnnz = idx.shape[1]
    idx = idx.tolist()
    val = val.tolist()
    x = x.squeeze(-1).tolist()
    
    for i in range(nnnz):
        current_idx = idx[1][i]
        if current_idx not in min_vals:
            min_vals[current_idx] = x[current_idx]

    
    for i in range(nnnz):
        current_row = idx[0][i]
        current_idx = idx[1][i]
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
                    new_sum += min_vals[b] * val[i]
                    # new_sum += min_vals[b] * A[row_id,b]
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
    
    # ts2 = eval_feas(A,res)
    # print(max(ts2),min(ts2))
    # quit()
    return res
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
    model.setObjective(vs.sum(), gp.GRB.MAXIMIZE)
    if A.is_sparse:
        # A=A.to_dense()
        Aind = A.indices().tolist()
        Aval = A.values().tolist()
        nc = A._nnz()
        current_row = -1
        buffer = []
        conss = []
        for i in range(nc):
            rind = Aind[0][i]
            cind = Aind[1][i]
            if rind != current_row:
                # new constriant, need to create constraint
                if len(buffer)!=0:
                    model.addConstr(gp.quicksum(j[0] * j[1] for j in buffer) <= 1.0)
                current_row = rind
                buffer =[]
            buffer.append([vs[cind],Aval[i]])
        if len(buffer)!=0:
            model.addConstr(gp.quicksum(j[0] * j[1] for j in buffer) <= 1.0)
    else:
        npy = A.numpy()
        model.addConstrs((gp.quicksum(vs[j] * npy[i,j] for j in range(n)) <= 1.0) for i in range(m))
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

def restore_feas_covering(A,x,y=None):
    
    x=torch.clamp(x,min=0.0,max=1.0)
    
    min_vals = {}
    spa = A.to_sparse()
    
    n = A.shape[1]
    m = A.shape[0]
    
    idx = spa.indices()
    val = spa.values()
    res = torch.zeros(x.shape)
    
    row_id = 0
    buffer = []
    buffer_sum = 0.0
    
    nnnz = idx.shape[1]
    idx = idx.tolist()
    val = val.tolist()
    x = x.squeeze(-1)[m:].tolist()
    for i in range(nnnz):
        current_idx = idx[1][i]
        if current_idx not in min_vals:
            min_vals[current_idx] = x[current_idx]
    
    for i in range(nnnz):
        current_row = idx[0][i]
        current_idx = idx[1][i]
        if row_id != current_row:
            if len(buffer)!=0 and buffer_sum<1.0 :
                # new constraint, need to deal with buffer
                print('New cons:',row_id,buffer_sum,len(buffer))
                for b in buffer:
                    newx1 = min_vals[b]/(buffer_sum)
                    min_vals[b] = max(min_vals[b], newx1)
                # print('        new row val: ',new_sum.item())
            buffer = []
            buffer_sum = 0.0
            row_id = current_row
            
        buffer.append(current_idx)    
        buffer_sum += val[i]*min_vals[current_idx]  
        print(val[i],min_vals[current_idx])  
        
    for key in min_vals:
        res[key] = max(min_vals[key],0.0)
        res[key] = min(min_vals[key],1.0)
    
    
    return res


@torch.no_grad()
def test(model, device, data_loader, loss_fn,mode = 'covering', file_order=[], folder = None):
    
    model.eval()
    y_true, y_pred = [], []
    test_loss = []
    
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
    
    flog = open(f'../logs/test_log_GEN_{folder}.log','w')
    
    with alive_bar(len(data_loader),title=f"Testing") as bar:
        for idx,g in enumerate(data_loader):
            st = time.time()
            g = g.to(device)
            logits = model(g, g.edata["feat"].to(device), g.ndata["feat"])
            inf_time = time.time() - st
            labels = g.ndata["label"].unsqueeze(-1)
            loss = loss_fn(logits, labels)
            test_loss.append(loss.item())
            
            st = f'/home/lxyang/git/GD-Net/data_{folder}/test/'
            fnm = f'{st}{file_order[idx]}'
            print(fnm)
            
            
            

            f = gzip.open(fnm,'rb')
            # A,v,c,sol,dual,obj = pickle.load(f)
            tar = pickle.load(f)
            Ak = tar[0]
            A=Ak.to(device)
            v = tar[1].to(device)
            c = tar[2].to(device)
            sol = tar[3]
            dual = tar[4]
            obj = tar[5]
            sol_time = tar[6]
            

            if len(tar)>=9:
                cost = tar[7]
                minA = tar[8]
            if not torch.is_tensor(A):
                A = torch.as_tensor(A,dtype=torch.float32).to(device)

            m = A.shape[0]

            x_gt = torch.as_tensor(sol,dtype=torch.float32).to(device)
            y_gt = torch.as_tensor(dual,dtype=torch.float32).to(device)
            f.close()
            
            n = A.shape[1]
            x = torch.ones((n,1)).to(device)
            y = torch.zeros((m,1)).to(device)
            
            x_res = logits
            
            st2 = time.time()
            if 'lp' in mode:
                x_res = restore_feas_LP(Ak,x_res)
                print('restored feasibility')
            elif 'covering' in mode or 'LSD' in mode:
                print("Restoring")
                x_res = restore_feas_covering(Ak,x_res)
                print('restored feasibility')
            elif 'maxflow' in mode:
                x_res = restore_feas_LP(Ak,x_res)
                print('restored feasibility')
            feas_time = time.time() - st2
            
            
            x_res = x_res.squeeze(-1)
            # if len(tar)>=9:
            #     x_res = x_res/minA
            #     for i in range(x_res.shape[0]):
            #         x_res[i] = x_res[i]/cost[i]

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
            
            
            model_grb = establish_grb(Ak,timelim=inf_time+feas_time)
            grb_obj, grb_time = solve_grb(Ak,x,y,model_grb)
            model_grb.Params.TimeLimit = grbtimelim
            grb_100_obj, grb_100_time = solve_grb_bestobj(Ak,x,y,model_grb,obj2)
            print(grb_obj, grb_time)

            # ws_obj, ws_time = solve_ws(A,x,y,model)
            # print(f'Warmstart: {ws_obj},{ws_time}')
            # quit()

            avg_grb_sametle += grb_obj
            avg_grb_beststop_time += grb_100_time
            
            
            print(f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{sol_time}\n  :::GRB:{grb_obj},{grb_100_time}')
            # print(x)
            st = f'Instance {fnm}::: ori obj:{obj}    pred obj:{obj2}   :::TIME: inf/feas/total/ori::{inf_time}/{feas_time}/{inf_time+feas_time}/{sol_time} :::GRB:{grb_obj},{grb_100_time}\n'
            flog.write(st)
            flog.flush()
            
            sum_obj+=obj
            sum_obj2+=obj2
            
            bar(1)
            
            
            
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
    return sum(test_loss) / len(test_loss)

    



def main():
    # check cuda
    
    ident = 'covering_1000_1000_600.0'
    # ident = 'lp_1000_1000_600.0'
    
    device = (
        f"cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    dataset_test, file_order = get_dataset_with_order(f'/home/lxyang/git/GD-Net/data_{ident}/test/',restart=False)
    node_feat_dim = dataset_test[0].ndata["feat"].size()[-1]
    edge_feat_dim = dataset_test[0].edata["feat"].size()[-1]
    # print( g,g.ndata["feat"].size())
    # print(g.ndata["feat"])
    # quit()
    n_classes = 1
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=64,
        out_dim=n_classes,
        num_layers=4,
        dropout=0.2,
        learn_beta=False,
    ).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()

    last_epoch = 0
    best_val_loss = None

    if not os.path.isdir('./model'):
        os.mkdir('./model')
    elif os.path.exists(f"./model/best_GEN_{ident}.mdl"):
        checkpoint = torch.load(f"./model/best_GEN_{ident}.mdl")
        model.load_state_dict(checkpoint['model'])
        if 'nepoch' in checkpoint:
            last_epoch=checkpoint['nepoch']
        best_val_loss=checkpoint['best_loss']
        print(f'Last best val loss gen:  {best_val_loss}')
        print('Model Loaded')
    print(count_parameters(model))
    # quit()
    print("---------- Training ----------")

    t1 = time.time()
    test_loss = test(model, device, dataset_test,  loss_fn,file_order = file_order, folder = ident, mode='lp')
    t2 = time.time()


    print('------------------------------------------------------------\n\n')



if __name__ == "__main__":
#     """
#     DeeperGCN Hyperparameters
#     """
    parser = argparse.ArgumentParser(description="DeeperGCN")
#     # training
#     parser.add_argument(
#         "--gpu", type=int, default=-1, help="GPU index, -1 for CPU."
#     )
#     parser.add_argument(
#         "--epochs", type=int, default=300, help="Number of epochs to train."
#     )
#     parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
#     parser.add_argument(
#         "--dropout", type=float, default=0.2, help="Dropout rate."
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=2048, help="Batch size."
#     )
#     # model
#     parser.add_argument(
#         "--num-layers", type=int, default=7, help="Number of GNN layers."
#     )
#     parser.add_argument(
#         "--hid-dim", type=int, default=256, help="Hidden channel size."
#     )
#     # learnable parameters in aggr
    args = parser.parse_args()

    print(f'Starting Testing GEN')
    
#     print(args)

    main()