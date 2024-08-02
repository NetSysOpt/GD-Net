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
import dgl


def from_bipart(A,ndim):
    n = A.shape[1]
    m = A.shape[0]
    
    indx = A.nonzero()
    indx_tp = A.nonzero(as_tuple=True)
    vals = A[indx_tp]
    vals = torch.cat((vals,vals),0)
    
    indx[:,1] += m
    indx_rev = torch.index_select(indx, 1, torch.LongTensor([1,0]))
    new_indx = torch.cat((indx,indx_rev),0)
    indx_rev = None
    return new_indx, vals   

def generate_csv_from_file(fnm,ndim=4,mode=0):
    
    f = gzip.open(fnm,'rb')
    tar = pickle.load(f)
    A = tar[0]
    bol = [False,False,False]
    bol[mode]=True
    sol = tar[3]
    dual = tar[4]
    n = A.shape[1]
    m = A.shape[0]
    f.close()
    
    new_indx, vals = from_bipart(A,ndim)
    # generate edge.csv
    f=open('./tmp_csv/edges.csv','w')
    st = f'src_id,dst_id,train_mask,val_mask,test_mask,feat\n'
    f.write(st)
    for id_idx,ele in enumerate(new_indx):
        src = ele[0].item()
        dst = ele[1].item()
        val = vals[id_idx].item()
        st = f'{src},{dst},{bol[0]},{bol[1]},{bol[2]},\"{val}\"\n'
        f.write(st)
    f.close()
    
    f=open('./tmp_csv/nodes.csv','w')
    st = f'node_id,label,train_mask,val_mask,test_mask,feat\n'
    f.write(st)
    for i in range(m):
        val = dual[i]
        st = f'{i},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0\"\n'
        f.write(st)
    for i in range(n):
        val = sol[i]
        st = f'{i+m},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0\"\n'
        f.write(st)
        
    f.close()
    
    
    
    