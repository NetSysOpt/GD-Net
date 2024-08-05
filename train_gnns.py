from model import *
import torch 
import gzip
import pickle
import os
import random
from alive_progress import alive_bar
import sys
sys.path.insert(1, './gcns/models')
from gcnconv import GCNConv
from hetero_gnn import *


random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

conv_type = 'gcnconv'
mdl = BipartiteHeteroGNN(conv=conv_type,
                            in_shape=2,
                            pe_dim=0,
                            hid_dim=64,
                            num_conv_layers=4,
                            num_pred_layers=1,
                            num_mlp_layers=1,
                            dropout=0,
                            share_conv_weight='false',
                            share_lin_weight='false',
                            use_norm='true',
                            use_res='false').to(device)

parm_num = count_parameters(mdl)

print(f'Number of parameters:: {parm_num}')

ident = "lp"
ident = "lp_15_15"
# ident = "lp_75_75"
# ident = "lp_25_25_60.0"
# ident = "lp_35_35_60.0"
# ident = "lp_45_45_60.0"
# ident = "lp_45_45_60.0"
ident = "lp_500_500"


ident = "IS_10"
ident = "IS_50"
ident = "IS_500"

ident = "CA_20_20"
ident = "CA_50_50"
# ident = "CA_300_300"

ident = 'covering_15_15_60.0'
# ident = 'covering_75_75_60.0'
ident = 'covering_1000_1000_60.0'
ident = "LSD_50"
ident = "LSD_500"
ident = "LSD_1000"


ident = "IS_1000"
ident = "lp_1000_1000_60.0"
ident = 'covering_500_500_600.0'
ident = 'lp_500_500_600.0'
ident = 'maxflow_1000_1000_600.0'
idf = f"data_{ident}"

flist_train = os.listdir(f'./{idf}/train')
flist_valid = os.listdir(f'./{idf}/valid')[:100]

best_loss = 1e+20
# mdl = GCNConv(2, 1, 180, 4, None, in_place = True)
# mdl = framework_model3(2,2,64,4)
last_epoch=0
if os.path.exists(f"./model/best_gcnconv_{ident}.mdl"):
    checkpoint = torch.load(f"./model/best_gcnconv_{ident}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-4)

max_epoch = 10000

flog = open(f'./logs/train_log_fixed_GCN_{ident}.log','w')

def to_full_adj(A,x,y,c=None):
    n = A.shape[1]
    m = A.shape[0]
    if c is None:
        c = torch.ones((n,1))
    
    if not A.is_sparse:
        A = A.to_sparse()
        
        
    aind = A.indices()
    aval = A.values()
    
    A2= torch.transpose(A,0,1).coalesce()
    aind2 = A2.indices()
    aval2 = A2.values()
    
    for i in range(aval.shape[0]):
        aind[0,i] += n
    for i in range(aval.shape[0]):
        aind2[1,i] += m
        
    aval = torch.cat((aval,aval2),dim = -1)
    aind = torch.cat((aind,aind2),dim = 1 )
    # need to add objective
    obj_coef = [[],[]]
    obj_vals = []
    for i in range(n):
        obj_coef[0].append(i)
        obj_coef[1].append(n+m)
        obj_vals.append(c[i])
        obj_coef[0].append(n+m)
        obj_coef[1].append(i)
        obj_vals.append(c[i])
    obj_coef = torch.as_tensor(obj_coef)
    obj_vals = torch.as_tensor(obj_vals)
    aind = torch.cat((aind,obj_coef),dim = 1 )
    aval = torch.cat((aval,obj_vals),dim = -1)
            
    resA = torch.sparse_coo_tensor(aind,aval,size=[n+m,n+m]).coalesce()
    
    ob = torch.tensor([1.0,1.0])
    ob = ob.unsqueeze(0)
    x = torch.cat((x,y,ob),dim=0)


    
    return resA,x



def to_full_adj_nonsq(A,x,y,c=None):
    n = A.shape[1]
    m = A.shape[0]
    if c is None:
        c = torch.ones((n,1))
    
    if not A.is_sparse:
        A = A.to_sparse()
        
        
    aind = A.indices().to(device)
    aval = A.values().to(device)
    
    # need to add objective
    obj_coef = [[],[]]
    obj_vals = []
    for i in range(n):
        obj_coef[0].append(m)
        obj_coef[1].append(i)
        obj_vals.append(c[i])
    obj_coef = torch.as_tensor(obj_coef).to(device)
    obj_vals = torch.as_tensor(obj_vals).to(device)
    aind = torch.cat((aind,obj_coef),dim = 1 )
    aval = torch.cat((aval,obj_vals),dim = -1)
            
    resA = torch.sparse_coo_tensor(aind,aval,size=[m,n]).coalesce()
    
    ob = torch.tensor([1.0,1.0])
    ob = ob.unsqueeze(0)
    # x = [torch.cat((y,ob),dim=0),x]
    x = [y,x]


    
    return resA,x

eps=0.2

for epoch in range(last_epoch, max_epoch):
    avg_loss=[0,0,0]
    random.shuffle(flist_train)
    with alive_bar(len(flist_train),title=f"GNN Training Epoch:{epoch}   task:{ident}") as bar:
        for fnm in flist_train:
            # train
            #  reading
            f = gzip.open(f'./data_{ident}/train/{fnm}','rb')
            tar = pickle.load(f)
            A=tar[0].to(device)
            v=tar[1].to(device)
            c=tar[2].to(device)
            sol=tar[3]
            dual=tar[4]
            obj=tar[5]
            A = torch.as_tensor(A,dtype=torch.float32).to(device)
            cost = None


            if len(tar)>=9:
                cost = torch.as_tensor(tar[7])
                minA = torch.as_tensor(tar[8])

            A,x = to_full_adj_nonsq(A,v,c,cost)

            m = A.shape[0]

            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            
            #  apply gradient 
            optimizer.zero_grad()

            n = A.shape[1]
            # x = torch.ones((n,1))
            # y = torch.zeros((m,1))
            # x = torch.as_tensor(v,dtype=torch.float32)
            # y = torch.as_tensor(c,dtype=torch.float32)
            x_dict = {}
            x_dict['cons'] = x[0]
            x_dict['vals'] = x[1]
            AT = torch.transpose(A,0,1).coalesce()
            ind_dict = {}
            val_dict = {}
            ind_dict[('vals','','cons')] = AT.indices()
            ind_dict[('cons','','vals')] = A.indices()
            val_dict[('vals','','cons')] = AT.values().unsqueeze(-1)
            val_dict[('cons','','vals')] = A.values().unsqueeze(-1)

            data = [x_dict,ind_dict,val_dict]
            # print(x[0])
            # quit()
            # print(x[0].shape,x[1].shape,x_gt.shape,y_gt.shape)
            x,_ = mdl(data)
            x = x[:,-1].unsqueeze(-1)
            
            x_gt = x_gt.unsqueeze(-1).to(device)
            loss_x = loss_func(x, x_gt)
            avg_loss[0] += loss_x.item()
            print(loss_x.item(),torch.sum(x).item())
            # loss_y = loss_func(y, y_gt)
            # avg_loss[1] += loss_y.item()
            # loss = loss_x+loss_y
            loss = loss_x
            # avg_loss[2] += loss.item()
            loss.backward()
            optimizer.step()
            bar()
    avg_loss[0] /= round(len(flist_train),2)
    avg_loss[1] /= round(len(flist_train),2)
    avg_loss[2] /= round(len(flist_train),2)
    # print(f'Epoch {epoch} Train:::: primal loss:{avg_loss[0]}, dual loss:{avg_loss[1]}, total loss:{avg_loss[2]}')
    print(f'Epoch {epoch} Train:::: primal loss:{avg_loss[0]}')
    st = f'{avg_loss[0]} '
    flog.write(st)



    avg_loss=[0,0,0]
    with alive_bar(len(flist_valid),title=f"Valid Epoch:{epoch}") as bar:
        for fnm in flist_valid:
            # valid
            #  reading
            f = gzip.open(f'./data_{ident}/valid/{fnm}','rb')
            # A,v,c,sol,dual,obj = pickle.load(f)
            tar = pickle.load(f)
            A=tar[0].to(device)
            v=tar[1].to(device)
            c=tar[2].to(device)
            sol=tar[3]
            dual=tar[4]
            obj=tar[5] 
            A = torch.as_tensor(A,dtype=torch.float32).to(device)
            cost = None


            if len(tar)>=9:
                cost = torch.as_tensor(tar[7])
                minA = torch.as_tensor(tar[8])

            A,x = to_full_adj_nonsq(A,v,c,cost)


            m = A.shape[0]

            x_gt = torch.as_tensor(sol,dtype=torch.float32).to(device)
            y_gt = torch.as_tensor(dual,dtype=torch.float32).to(device)
            f.close()
            
            n = A.shape[1]
            # x = torch.ones((n,1))
            # y = torch.zeros((m,1))
            #  obtain loss
            x_dict = {}
            x_dict['cons'] = x[0]
            x_dict['vals'] = x[1]
            AT = torch.transpose(A,0,1).coalesce()
            ind_dict = {}
            val_dict = {}
            ind_dict[('vals','','cons')] = AT.indices()
            ind_dict[('cons','','vals')] = A.indices()
            val_dict[('vals','','cons')] = AT.values().unsqueeze(-1)
            val_dict[('cons','','vals')] = A.values().unsqueeze(-1)

            data = [x_dict,ind_dict,val_dict]
            # print(x[0])
            # quit()
            # print(x[0].shape,x[1].shape,x_gt.shape,y_gt.shape)
            x,_ = mdl(data)
            x = x[:,-1].unsqueeze(-1)

            # x = mdl(x,A.indices(),A.values().unsqueeze(-1))
            x_gt = x_gt.unsqueeze(-1)
            loss_x = loss_func(x, x_gt)
            avg_loss[0] += loss_x.item()
            # loss_y = loss_func(y, y_gt)
            # avg_loss[1] += loss_y.item()
            # loss = loss_x+loss_y
            # avg_loss[2] += loss.item()
            bar()
    avg_loss[0] /= round(len(flist_valid),2)
    avg_loss[1] /= round(len(flist_valid),2)
    avg_loss[2] /= round(len(flist_valid),2)
    # print(f'Epoch {epoch} Valid:::: primal loss:{avg_loss[0]}, dual loss:{avg_loss[1]}, total loss:{avg_loss[2]}')
    print(f'Epoch {epoch} Valid:::: primal loss:{avg_loss[0]}')
    st = f'{avg_loss[0]}\n'
    flog.write(st)
    if best_loss > avg_loss[0]:
        best_loss = avg_loss[0]
        
        state={'model':mdl.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':best_loss,'nepoch':epoch}
        torch.save(state,f'./model/best_gcnconv_{ident}.mdl')

        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()