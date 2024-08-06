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
import os
import shutil
from alive_progress import alive_bar

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

def get_graph(fnm,fn,fsec,dir_tar):
    generate_csv_from_file(fnm,ndim=2,mode=0)
    graph = dgl.data.CSVDataset(dir_tar)[0]
    fout = gzip.open(f'{fsec}/{fn}','wb')
    pickle.dump(graph,fout)
    fout.close()
    return graph
        
def get_dataset(fdir,restart=False):
    full_ds = []
    ident = fdir.split('/')
    ident = [x for x in ident if x!='']
    froot = f'./{ident[-2]}'
    fsec = f'./{ident[-2]}/{ident[-1]}'
    if os.path.isdir(fsec) and not restart:
        # already extracted
        print('TODO::')
        ssss = os.listdir(fsec)
        with alive_bar(len(ssss),title=f"Processing {len(ssss)} instances") as bar:
            for fn in ssss:
                fout = gzip.open(f'{fsec}/{fn}','rb')
                graph = pickle.load(fout)
                fout.close()
                full_ds.append(graph)
                bar()
    else:
        # not extracted
        if not os.path.isdir(froot):
            os.mkdir(froot)
            os.mkdir(fsec)
        else:
            os.mkdir(fsec)
        ssss = os.listdir(fdir)
        with alive_bar(len(ssss),title=f"Processing {len(ssss)} instances") as bar:
            for fn in ssss:
                dir_tar = './tmp_csv'
                if os.path.isdir(f"{dir_tar}/tmp_csv"):
                    shutil.rmtree(f"{dir_tar}/tmp_csv")
                fnm = fdir+fn
                graph = get_graph(fnm,fn,fsec,dir_tar)
                full_ds.append(graph)

                print(f'processed {fnm}')
                bar(1)
    return full_ds
    

        
def get_dataset_with_order(fdir,restart=False):
    full_ds = []
    ident = fdir.split('/')
    ident = [x for x in ident if x!='']
    froot = f'./{ident[-2]}'
    fsec = f'./{ident[-2]}/{ident[-1]}'
    fnms = []
    if os.path.isdir(fsec) and not restart:
        # already extracted
        print('TODO::')
        ssss = os.listdir(fsec)
        with alive_bar(len(ssss),title=f"Processing {len(ssss)} instances") as bar:
            for fn in ssss:
                fout = gzip.open(f'{fsec}/{fn}','rb')
                graph = pickle.load(fout)
                fout.close()
                full_ds.append(graph)
                fnms.append(fn)
                bar()
    else:
        # not extracted
        if not os.path.isdir(froot):
            os.mkdir(froot)
            os.mkdir(fsec)
        else:
            os.mkdir(fsec)
        ssss = os.listdir(fdir)
        with alive_bar(len(ssss),title=f"Processing {len(ssss)} instances") as bar:
            for fn in ssss:
                dir_tar = './tmp_csv'
                if os.path.isdir(f"{dir_tar}/tmp_csv"):
                    shutil.rmtree(f"{dir_tar}/tmp_csv")
                fnm = fdir+fn
                graph = get_graph(fnm,fn,fsec,dir_tar)
                full_ds.append(graph)
                fnms.append(fn)

                print(f'processed {fnm}')
                bar(1)
    return full_ds,fnms

# pool = multiprocessing.Pool(processes=4) 
# for i in range(train_files):
#     pool.apply_async(create_and_save_mf, args=(f"./data_maxflow_{n}_{m}_{pillar}/train/prob_{i}.pkl",n,m,p,mode,))
# pool.close()
# pool.join()


def get_dataset_wholeset(fdir,restart=False):
    full_ds = []
    ident = fdir.split('/')
    ident = [x for x in ident if x!='']
    froot = f'./{ident[-2]}'
    fsec = f'./{ident[-2]}/{ident[-1]}'
    if fdir[-1]!='/':
        fdir+='/'
    if os.path.isfile(f'{fdir}pkl.pkl') and not restart:
        # already extracted
        print('TODO::')
        fout = gzip.open(f'{fsec}/pkl.pkl','rb')
        full_ds = pickle.load(fout)
        fout.close()
    else:
        # not extracted
        if not os.path.isdir(froot):
            os.mkdir(froot)
            os.mkdir(fsec)
        elif not os.path.isdir(fsec):
            os.mkdir(fsec)

        if os.path.isdir("./tmp_csv_whole/tmp_csv"):
            shutil.rmtree("./tmp_csv_whole/tmp_csv")
        generate_csv_from_files(fdir,ndim=2,mode=0)
        full_ds = dgl.data.CSVDataset('./tmp_csv_whole')
        fout = gzip.open(f'{fsec}/pkl.pkl','wb')
        pickle.dump(full_ds,fout)
        fout.close()
    return full_ds


def generate_csv_from_files(fdir,ndim=4,mode=0):
    
    if not os.path.isdir('./tmp_csv_whole'):
        os.mkdir('./tmp_csv_whole')
        fout=open('./tmp_csv_whole/meta.yaml','w')
        st = 'dataset_name: tmp_csv_whole\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv\ngraph_data:\n  file_name: graphs.csv'
        fout.write(st)
        fout.close()

    if fdir[-1]!='/':
        fdir += '/'

    gid = 0
    flsts = os.listdir(fdir)
    f_edg=open('./tmp_csv_whole/edges.csv','w')
    st = f'graph_id,src_id,dst_id,train_mask,val_mask,test_mask,feat\n'
    f_edg.write(st)
    f_node=open('./tmp_csv_whole/nodes.csv','w')
    st = f'graph_id,node_id,label,train_mask,val_mask,test_mask,feat\n'
    f_node.write(st)
    with alive_bar(len(flsts),title=f"Processing {len(flsts)} instances") as bar:
        for fn in flsts:
            fnm = f'{fdir}{fn}'
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
            for id_idx,ele in enumerate(new_indx):
                src = ele[0].item()
                dst = ele[1].item()
                val = vals[id_idx].item()
                st = f'{gid},{src},{dst},{bol[0]},{bol[1]},{bol[2]},\"{val}, 1.0\"\n'
                f_edg.write(st)
            f_edg.flush()
            for i in range(m):
                val = dual[i]
                st = f'{gid},{i},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0, 1.0\"\n'
                f_node.write(st)
            for i in range(n):
                val = sol[i]
                st = f'{gid},{i+m},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0, 1.0\"\n'
                f_node.write(st)
            f_node.flush()


            gid+=1
            bar()
    
    f_node.close()
    f_edg.close()

    f=open('./tmp_csv_whole/graphs.csv','w')
    st = f'graph_id\n'
    f.write(st)
    for k in range(gid):
        st = f'{k}\n'
        f.write(st)
    f.close()
    
    




def generate_csv_from_file(fnm,ndim=4,mode=0):
    
    if not os.path.isdir('./tmp_csv'):
        os.mkdir('./tmp_csv')
        fout=open('./tmp_csv/meta.yaml','w')
        st = 'dataset_name: tmp_csv\nedge_data:\n- file_name: edges.csv\nnode_data:\n- file_name: nodes.csv'
        fout.write(st)
        fout.close()

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
        st = f'{src},{dst},{bol[0]},{bol[1]},{bol[2]},\"{val}, 0.0\"\n'
        f.write(st)
    f.close()
    
    f=open('./tmp_csv/nodes.csv','w')
    st = f'node_id,label,train_mask,val_mask,test_mask,feat\n'
    f.write(st)
    for i in range(m):
        val = dual[i]
        st = f'{i},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0, 0.0\"\n'
        f.write(st)
    for i in range(n):
        val = sol[i]
        st = f'{i+m},{val},{bol[0]},{bol[1]},{bol[2]},\"0.0, 0.0\"\n'
        f.write(st)
        
    f.close()
    
    
    
    