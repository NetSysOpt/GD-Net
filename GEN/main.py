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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, data_loader, opt, loss_fn, epoch = 0, file_order=[], folder = None):
    model.train()
    train_loss = []
    parm_num = count_parameters(model)
    with alive_bar(len(data_loader),title=f"GEN Training Epoch:{epoch}    nParm:{parm_num}  ") as bar:
        for idx,g in enumerate(data_loader):
            st = f'/home/lxyang/git/GD-Net/data_{folder}/train/'
            fnm = f'{st}{file_order[idx]}'
            f = gzip.open(fnm,'rb')
            # A,v,c,sol,dual,obj = pickle.load(f)
            tar = pickle.load(f)
            Ak = tar[0]
            ncons = Ak.shape[0]
            f.close()
            
            
            # print(g.ndata['feat'])
            g = g.to(device)
            logits = model(g, g.edata["feat"].to(device), g.ndata["feat"])
            labels = g.ndata["label"].unsqueeze(-1)
            loss = loss_fn(logits[ncons:], labels[ncons:])
            train_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            bar()

    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader, loss_fn,epoch, file_order=[], folder = None):
    
    model.eval()
    y_true, y_pred = [], []
    train_loss = []
    with alive_bar(len(data_loader),title=f"GEN Valid Epoch:{epoch} ") as bar:
        for idx,g in enumerate(data_loader):
            st = f'/home/lxyang/git/GD-Net/data_{folder}/valid/'
            fnm = f'{st}{file_order[idx]}'
            f = gzip.open(fnm,'rb')
            # A,v,c,sol,dual,obj = pickle.load(f)
            tar = pickle.load(f)
            Ak = tar[0]
            ncons = Ak.shape[0]
            f.close()
            
            g = g.to(device)
            logits = model(g, g.edata["feat"].to(device), g.ndata["feat"])
            labels = g.ndata["label"].unsqueeze(-1)
            loss = loss_fn(logits[ncons:], labels[ncons:])
            train_loss.append(loss.item())
            bar(1)
    return sum(train_loss) / len(train_loss)

class new_data:
    def __init__(self):
        self.ndata = {}
        self.ndata["feat"] = None
        self.edata = {}
        self.edata["feat"] = None
        

    



def main(cont=False,restart=False):
    # check cuda
    
    ident = 'covering_1000_1000_600.0'
    # ident = 'lp_1000_1000_600.0'
    
    device = (
        f"cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    dataset_train, train_fnms = get_dataset_with_order(f'/home/lxyang/git/GD-Net/data_{ident}/train/',restart)
    dataset_valid, valid_fnms = get_dataset_with_order(f'/home/lxyang/git/GD-Net/data_{ident}/valid/',restart)
    node_feat_dim = dataset_train[0].ndata["feat"].size()[-1]
    edge_feat_dim = dataset_train[0].edata["feat"].size()[-1]
    # print( g,g.ndata["feat"].size())
    # print(g.ndata["feat"])
    # quit()
    n_classes = 1
    model = DeeperGCN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hid_dim=64,
        out_dim=n_classes,
        num_layers=8,
        dropout=0.2,
        learn_beta=False,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    last_epoch = 0
    best_val_loss = None

    if not os.path.isdir('./model'):
        os.mkdir('./model')
    elif os.path.exists(f"./model/best_GEN_{ident}.mdl") and cont:
        checkpoint = torch.load(f"./model/best_GEN_{ident}.mdl")
        model.load_state_dict(checkpoint['model'])
        if 'nepoch' in checkpoint:
            last_epoch=checkpoint['nepoch']
        best_val_loss=checkpoint['best_loss']
        print(f'Last best val loss gen:  {best_val_loss}')
        print('Model Loaded')

    print("---------- Training ----------")

    for i in range(last_epoch,100000):
        t1 = time.time()
        train_loss = train(model, device, dataset_train, opt, loss_fn,i,file_order = train_fnms, folder = ident)
        t2 = time.time()

        t1 = time.time()
        valid_loss = test(model, device, dataset_valid, loss_fn,i,file_order = valid_fnms, folder = ident)
        t2 = time.time()

        print(f'Training epoch{i} ----- loss:{train_loss}')
        print(f'Training epoch{i} ----- loss:{valid_loss}')

        if best_val_loss is None or best_val_loss>valid_loss:
            best_val_loss = valid_loss
            
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'best_loss':best_val_loss,'nepoch':i}
            torch.save(state,f'./model/best_GEN_{ident}.mdl')
            print(f'Saving new best model with valid loss: {best_val_loss}')

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
    parser.add_argument('-c',"--cont", action="store_true")
    parser.add_argument('-r',"--restart", action="store_true")

    args = parser.parse_args()
    print(f'Starting training GEN, continue?{args.cont}, restart?{args.restart}')
    
#     print(args)

    main(args.cont,args.restart)