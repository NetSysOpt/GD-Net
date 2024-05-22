from model import *
import torch 
import gzip
import pickle
import os
import random
from alive_progress import alive_bar

random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ident = "covering"
# ident = "covering_20_20_60.0"
ident = "covering_15_15_60.0"
ident = "covering_75_75_60.0"
ident = "covering_500_500_60.0"
ident = "covering_1000_1000_60.0"
ident = "LSD_50"
ident = "LSD_500"
ident = "covering_500_500_60.0"
ident = "covering_75_75_60.0"
# ident = "LSD_1000"
idf = f"data_{ident}"
print("training",ident)

lr1 = 1e-3



nfeat = 5
model_type = 'dchannel'


# nfeat = 1
# model_type = ''

other=f'{model_type}'
if nfeat!=1:
    other += f'_feat{nfeat}'

flist_train = os.listdir(f'./{idf}/train')
flist_valid = os.listdir(f'./{idf}/valid')[:100]

best_loss = 1e+20
# mdl = framework_model1dim(4,64,nfeat)
mdl = framework_model1dim_covering(4,64,nfeat)
if 'dchannel' == model_type:
    print('!!!!!!!!!!USING Dchannel model, method 2')
    mdl = framework_model1dim_covering(4,64,nfeat,mode=model_type)
    
parm_num = count_parameters(mdl)

print(f'Number of parameters:: {parm_num}')


# mdl = framework_model3(2,2,64,4)
last_epoch=0
if os.path.exists(f"./model/best_covering_{ident}{other}.mdl"):
    checkpoint = torch.load(f"./model/best_covering_{ident}{other}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=lr1)

max_epoch = 10000

flog = open(f'./logs/train_log_covering_{ident}{other}.log','w')

eps=0.2

for epoch in range(last_epoch, max_epoch):
    avg_loss=[0,0,0]
    random.shuffle(flist_train)
    with alive_bar(len(flist_train),title=f"Covering Training Epoch:{epoch}    nParm:{parm_num}  task:{ident}") as bar:
        for fnm in flist_train:
            # train
            #  reading
            f = gzip.open(f'./data_{ident}/train/{fnm}','rb')
            tar = pickle.load(f)
            A = tar[0]
            v = tar[1]
            c = tar[2]
            sol = tar[3]
            dual = tar[4]
            obj = tar[5]
            A = torch.as_tensor(A,dtype=torch.float32)
            amx = torch.max(A.to_dense())
            m = A.shape[0]
            mu = 1/eps * torch.log(m*amx/eps)

            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            #  apply gradient 
            optimizer.zero_grad()
            n = A.shape[1]

            # x = torch.ones((n,1))
            x = torch.ones((n,1))
            y = torch.zeros((m,1))
            x,y = mdl(A,x,y,mu)

            
            # for name,param in mdl.named_parameters():
            #     if param.requires_grad and 't2' in name:
            #         print(name)
            #         print(param)
            # # print(x)
            # # print(fnm, mu)
            # quit()

            x_gt = x_gt.unsqueeze(-1)
            loss_x = loss_func(x, x_gt)
            avg_loss[0] += loss_x.item()
            print(loss_x.item())
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
            A = tar[0]
            v = tar[1]
            c = tar[2]
            sol = tar[3]
            dual = tar[4]
            obj = tar[5]
            A = torch.as_tensor(A,dtype=torch.float32)

            # amx = torch.max(A)
            amx = torch.max(A.to_dense())
            m = A.shape[0]
            mu = 1/eps * torch.log(m*amx/eps)

            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            #  obtain loss

            n = A.shape[1]
            # x = torch.ones((n,1))
            x = torch.ones((n,1))
            y = torch.zeros((m,1))


            x_gt = x_gt.unsqueeze(-1)
            x,y = mdl(A,x,y,mu)
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
        torch.save(state,f'./model/best_covering_{ident}{other}.mdl')

        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()