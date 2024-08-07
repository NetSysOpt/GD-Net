import os
import copy
import gzip
import time
import torch
import pickle
import random
class bipart_adj:
    def __init__(self):
        self.edges = {}
        # src is -1
        self.edges[-1] = {}
        # dst is -2
        # right side nodes are n+i

        self.n = None
        self.m = None

    def init_src_dst(self,src_cap,dst_cap,n,m):
        for idx,val in enumerate(src_cap):
            self.edges[-1][idx] = val
        for idx,val in enumerate(dst_cap):
            self.edges[idx+n] = {}
            if val==0:
                continue
            self.edges[idx+n][-2] = val
        self.n = n
        self.m = m
    
    def insert_node(self,x,y,val):
        if x not in self.edges:
            self.edges[x] = {}
        self.edges[x][y+self.n] = val

    def get_adjs(self,x):
        if x not in self.edges:
            return {}
        return self.edges[x]
    
    def get_cap(self,x,y):
        # print(self.edges[x],y+self.n)
        # if x != -1:
        #     if x in self.edges and y+self.n in self.edges[x] and self.edges[x][y+self.n]>0:
        #         return self.edges[x][y+self.n]
        #     else:
        #         return None
        # else:
        #     if y in self.edges[x] and self.edges[x][y]>0:
        #         return self.edges[x][y]
        #     else:
        #         return None
        if x in self.edges and y in self.edges[x] and self.edges[x][y]>0:
            return self.edges[x][y]
        else:
            return None

    def alter_cap(self,x,y,cap):
        if x in self.edges and y in self.edges[x]:
            self.edges[x][y] += cap
        elif cap == 0:
            return 
        elif x in self.edges:
            self.edges[x][y] = cap
        else:
            self.edges[x] = {}
            self.edges[x][y] = cap
        if self.edges[x][y] == 0:
            self.edges[x].pop(y)
            
    def get_total_flow(self):
        sums = 0.0
        for idx in self.edges:
            if idx<0:
                continue
            if -2 not in self.edges[idx]:
                continue
            sums += self.edges[idx][-2]
        return sums

# since this is a pitartite graph, we use dfs for better efficiency
def dfs(current_A,src=-1,snk=-2):
    cap = 0.0
    stack = [src]
    parent = [None]
    ptr = None
    visited = set()
    history = []
    while len(stack)>0:
        ptr = stack.pop()
        pare = parent.pop()
        if ptr == snk:
            history.append(-2)
            min_cap = 1e+20
            print(history)
            for i in range(len(history)-1):
                min_cap = min(min_cap, current_A.get_cap(history[i],history[i+1]))
            return history, min_cap
        visited.add(ptr)
        tars = current_A.get_adjs(ptr)
        # print(f'ptr:{ptr}  {tars}  {pare}  ')
        while len(history)>0 and pare!=history[-1]:
            history.pop()
        if len(tars)==0:
            continue
        history.append(ptr)
        for tar in tars:
            if tars[tar] > 0 and tar not in visited:
                stack.append(tar)
                parent.append(ptr)
                # print(f'added {tar},   {tars[tar]}')
        
    return None, None
def dfs_rec(current_A,history,ptr=-1,snk=-2):
    
    if ptr == snk:
        return [],1e+20
    tars = current_A.get_adjs(ptr)
    if len(tars)==0:
        return None,None
    history.add(ptr)
    for tar in tars:
        if tars[tar] > 0 and tar not in history:
            path, flow = dfs_rec(current_A,history,ptr=tar,snk=-2)
            if flow is None:
                continue
            path.insert(0,tar)
            return path, min(tars[tar],flow)
    history.remove(ptr)
    
    
    return None,None


# Ford Fulkerson implementation to solve the bipartite maxflow prob
def ff(A,stopper = None):
    # for edg in A.edges:
    #     print(edg,A.edges[edg])
    #     input()
    # quit()
    st = time.time()
    A_rev = bipart_adj()
    iters = 0
    
    total_flow = 0.0
    while True:
        hist = set()
        path, flow = dfs_rec(A,hist)
        if flow is None:
            # finished FF, no more possible flows
            print(f'No more flow')
            break
        total_flow += flow
        if stopper is not None and stopper<=total_flow:
            break
        path.insert(0,-1)
        if iters%20000 == 0:
            # print(f'Iter{iters}::::Found a flow with {flow}: {path}')
            print(f'Iter{iters}::::Current flow: {total_flow}')
        # otherwise
        # need to apply this flow to the map
        for i in range(len(path)-1):
            A.alter_cap(path[i],path[i+1],-flow)
            A.alter_cap(path[i+1],path[i],flow)
            A_rev.alter_cap(path[i],path[i+1],flow)
        iters+=1
        
                
                
    tf = A_rev.get_total_flow()
    ttime =round(time.time()-st,4)
    
    if stopper is None:
        print(f'Max Flow: {tf}')
    else:
        print(f'Current Flow: {tf} / {stopper}')
        
    print(f'   in {ttime}s / {iters} iters')
    return tf,ttime,iters



def run(fnm,logfile=None):
    f = gzip.open(fnm,'rb')
    tar = pickle.load(f)
    Ak = tar[0]
    nv = Ak.shape[0]//2
    f.close()
    Aindx = Ak.indices().tolist()
    Avals = Ak.values()
    Avals = Avals
    Avals = Avals.tolist()
    nnz = len(Avals)
    print(nnz,nv)
    
    fnm2 = fnm.replace('maxflow','maxflow_info')
    f = gzip.open(fnm2,'rb')
    package = pickle.load(f)
    v_bounds=package[0]
    src_bound=package[1]
    dst_bound=package[2]
    f.close()
    
    print(len(v_bounds),len(Avals))
    
    A = bipart_adj()
    srcs = src_bound
    dsts = dst_bound
    A.init_src_dst(srcs,dsts,nv,nv)
    
    # edge_maps = {}
    # edge_vals = {}
    # for idx in range(nnz):
    #     idx_edge = Aindx[1][idx]
    #     idx_node = Aindx[0][idx]
    #     if idx_edge not in edge_maps:
    #         edge_maps[idx_edge] = []
    #     if idx_edge not in edge_vals:
    #         edge_vals[idx_edge] = []
    #     edge_vals[idx_edge] = Avals[idx]    
    #     edge_maps[idx_edge].append(idx_node)
        
    # for idx in edge_maps:
    #     src = edge_maps[idx][0]
    #     dst = edge_maps[idx][1]
    #     val = edge_vals[idx]
    #     A.insert_node(src,dst-nv,val)
    #     # print(f'{src}->{dst}   {val}')
    
    for edg in v_bounds:
        src = edg[0]
        dst = edg[1]
        val = edg[2]
        A.insert_node(src,dst,val)
        
    
    pred_obj=None
    if logfile is not None:
        with open(logfile,'r') as f:
            for line in f:
                if 'Instance' in line:
                    line = line.replace('\n','').split(' ')
                    line = [x for x in line if x!='']
                    pred_obj = float(line[5].split(':')[-1])
                    fz = line[1].replace(':','')
                    our_time = float(line[7].split(':')[-1].split('/')[2])
                    if fz in fnm:
                        print(fz,pred_obj,str(our_time)+'s')
                        break
    return ff(A,pred_obj)
    
    
print(run('/home/lxyang/git/GD-Net/data_maxflow_600_600_600.0/test/prob_0.pkl','/home/lxyang/git/GD-Net/logs/test_log_model3_maxflow_600_600_600.0x0dchannel_feat5.log'))
print(run('/home/lxyang/git/GD-Net/data_maxflow_1000_1000_600.0/test/prob_0.pkl','/home/lxyang/git/GD-Net/logs/test_log_model3_maxflow_1000_1000_600.0x0dchannel_feat5.log'))