from model import *
import torch 
import gzip
import pickle
import os
import random
from alive_progress import alive_bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ident = "lp"
idf = f"data_{ident}"

flist_train = os.listdir(f'./{idf}/train')
flist_valid = os.listdir(f'./{idf}/valid')

mdl = framework_learn_mu(2,2,64,4)


best_loss = 1e+20
last_epoch=0
if os.path.exists(f"./model/best_model_{ident}.mdl"):
    checkpoint = torch.load(f"./model/best_model_{ident}.mdl")
    mdl.load_state_dict(checkpoint['model'])
    if 'nepoch' in checkpoint:
        last_epoch=checkpoint['nepoch']
    best_loss=checkpoint['best_loss']
    print(f'Last best val loss gen:  {best_loss}')
    print('Model Loaded')


loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-4)

max_epoch = 10000

flog = open('./logs/train_log.log','w')


for epoch in range(last_epoch,max_epoch):
    avg_loss=[0,0,0]
    random.shuffle(flist_train)
    with alive_bar(len(flist_train),title=f"Training Epoch:{epoch}") as bar:
        for fnm in flist_train:
            # train
            #  reading
            f = gzip.open(f'./{idf}/train/{fnm}','rb')
            A,v,c,sol,dual,obj = pickle.load(f)
            A = torch.as_tensor(A,dtype=torch.float32)
            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            #  apply gradient 
            optimizer.zero_grad()
            x,y = mdl(A,x,y)
            loss_x = loss_func(x, x_gt)
            avg_loss[0] += loss_x.item()
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
            f = gzip.open(f'./{idf}/valid/{fnm}','rb')
            A,v,c,sol,dual,obj = pickle.load(f)
            A = torch.as_tensor(A,dtype=torch.float32)
            x = torch.as_tensor(v,dtype=torch.float32)
            y = torch.as_tensor(c,dtype=torch.float32)
            x_gt = torch.as_tensor(sol,dtype=torch.float32)
            y_gt = torch.as_tensor(dual,dtype=torch.float32)
            f.close()
            #  obtain loss
            x,y = mdl(A,x,y)
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
        torch.save(state,f'./model/best_model_{ident}.mdl')
        
        print(f'Saving new best model with valid loss: {best_loss}')

    flog.flush()


flog.close()