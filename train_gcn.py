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
# ident = "lp_15_15"
ident = "lp_75_75"
ident = "lp_25_25_60.0"
ident = "lp_35_35_60.0"
ident = "lp_45_45_60.0"
# ident = "lp_500_500"
idf = f"data_{ident}"

flist_train = os.listdir(f'./{idf}/train')
flist_valid = os.listdir(f'./{idf}/valid')

best_loss = 1e+20
mdl = GCN(2,2,64)
# mdl = framework_model3(2,2,64,4)
last_epoch=0
if os.path.exists(f"./model/best_GCN_{ident}.mdl"):
    checkpoint = torch.load(f"./model/best_GCN_{ident}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)

max_epoch = 10000

flog = open(f'./logs/train_log_fixed_GCN_{ident}.log','w')

eps=0.2

for epoch in range(last_epoch, max_epoch):
    avg_loss=[0,0,0]
    random.shuffle(flist_train)
    with alive_bar(len(flist_train),title=f"Training Epoch:{epoch}") as bar:
        for fnm in flist_train:
            # train
            #  reading
            f = gzip.open(f'./data_{ident}/train/{fnm}','rb')
            tar = pickle.load(f)
            A=tar[0]
            v=tar[1]
            c=tar[2]
            sol=tar[3]
            dual=tar[4]
            obj=tar[5] 
            A = torch.as_tensor(A,dtype=torch.float32)

            m = A.shape[0]

            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            #  apply gradient 
            optimizer.zero_grad()

            n = A.shape[1]
            # x = torch.ones((n,1))
            # y = torch.zeros((m,1))
            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            x = mdl(A,x,y)
            
            x_gt = x_gt.unsqueeze(-1)
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
            A=tar[0]
            v=tar[1]
            c=tar[2]
            sol=tar[3]
            dual=tar[4]
            obj=tar[5] 
            A = torch.as_tensor(A,dtype=torch.float32)

            m = A.shape[0]

            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            
            n = A.shape[1]
            # x = torch.ones((n,1))
            # y = torch.zeros((m,1))
            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            #  obtain loss

            x = mdl(A,x,y)
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
        torch.save(state,f'./model/best_GCN_{ident}.mdl')

        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()