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


ident = "lp"
# ident = "lp_20_20"
# ident = "lp_15_15"
ident = "lp_75_75"
# ident = "lp_500_500"

ident = "CA_20_20"
ident = "CA_50_50"
# ident = "CA_300_300"

# ident = "IS_10"
ident = "IS_500"
ident = "IS_1000"
ident = "IS_50"
# ident = "lp_75_75_60.0"
# ident = "lp_500_500_60.0"
ident = "lp_1000_1000_60.0"
eps=0.2





nfeat = 5
model_type = 'dchannel'




lrate = 1e-3
conts=True

lrate = 1e-4
conts=True

# lrate = 1e-6
# conts=True

idf = f"data_{ident}"


nfeat = 5
other='x0'+model_type
if nfeat!=1:
    other += f'_feat{nfeat}'
if eps!=0.2:
    other += f'_ep{eps}'

flist_train = os.listdir(f'./{idf}/train')
flist_valid = os.listdir(f'./{idf}/valid')[:100]

best_loss = 1e+20
mdl = framework_model1dim(4,64,nfeat,model_type)
parm_num = count_parameters(mdl)

print(f'Number of parameters:: {parm_num}')
# mdl = framework_model3(2,2,64,4)
last_epoch=0
if os.path.exists(f"./model/best_model3_{ident}{other}.mdl") and conts:
    checkpoint = torch.load(f"./model/best_model3_{ident}{other}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=lrate)

max_epoch = 10000

flog = open(f'./logs/train_log_fixed_mu_{ident}{other}.log','w')


for epoch in range(last_epoch, max_epoch):
    avg_loss=[0,0,0]
    random.shuffle(flist_train)
    with alive_bar(len(flist_train),title=f"Model3 Training Epoch:{epoch} task:{ident}") as bar:
        for fnm in flist_train:
            # train
            #  reading
            f = gzip.open(f'./data_{ident}/train/{fnm}','rb')
            # A,v,c,sol,dual,obj = pickle.load(f)
            tar = pickle.load(f)
            A = tar[0]
            v = tar[1]
            c = tar[2]
            sol = tar[3]
            dual = tar[4]
            obj = tar[5]

            if len(tar)>=9:
                cost = torch.as_tensor(tar[7])
                minA = torch.as_tensor(tar[8])

            A = torch.as_tensor(A,dtype=torch.float32)

            amx = None
            if A.is_sparse:
                amx = torch.max(A.values())
            else:
                amx = torch.max(A)
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
            x = torch.zeros((n,1))
            y = torch.zeros((m,1))
            x,y = mdl(A,x,y,mu)

            # check if need to restore val
            # if len(tar)>=9:
            #     x = x/minA
            #     for i in range(x.shape[0]):
            #         x[i] = x[i]/cost[i]


            
            # for name,param in mdl.named_parameters():
            #     if param.requires_grad and 't2' in name:
            #         print(name)
            #         print(param)
            # # print(x)
            # # print(fnm, mu)
            # quit()

            x_gt = x_gt.unsqueeze(-1)
            loss_x = loss_func(x, x_gt)
            # print(x.min(),x.max())
            # print(x_gt.min(),x_gt.max())
            # input()
            avg_loss[0] += loss_x.item()
            if len(tar)>=9:
                print(loss_x.item(),torch.sum(x).item()/minA.item(),obj)
            else:
                print(loss_x.item(),torch.sum(x).item(),obj)
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
    st = f'Epoch {epoch}::::{avg_loss[0]} '
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


            if len(tar)>=9:
                cost = tar[7]
                minA = tar[8]

            A = torch.as_tensor(A,dtype=torch.float32)

            amx = None
            if A.is_sparse:
                amx = torch.max(A.values())
            else:
                amx = torch.max(A)
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
            x = torch.zeros((n,1))
            y = torch.zeros((m,1))

            x_gt = x_gt.unsqueeze(-1)

            x,y = mdl(A,x,y,mu)
            # check if need to restore val
            # if len(tar)>=9:
            #     x = x/minA
            #     for i in range(x.shape[0]):
            #         x[i] = x[i]/cost[i]

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
        torch.save(state,f'./model/best_model3_{ident}{other}.mdl')

        st =f'Saving new best model with valid loss: {best_loss}\n'
        flog.write(st)
        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()