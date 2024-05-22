import torch
import torch.nn as nn
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def print_parameters(model):
    z = [p for p in model.parameters() if p.requires_grad]
    for zz in z:
        print(zz.numel(),zz.names)
    
class update_cycle_model3(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,channels=16, mode = 'ori'):
        super(update_cycle_model3,self).__init__()
        self.y_up = y_update_model3(x_size,feat_size)
        # for indx in range(n_layers):
        #     self.updates.append(x_update_model3(x_size, feat_size,feat_size))
        self.x_up = x_update_model3(x_size, feat_size,feat_size,channels, mode=mode)

        self.feat_size = feat_size
        
    def forward(self,A,x,y,mu):

        # compute AX
        y_new = self.y_up(A,x,mu)
        x_new = self.x_up(A,x,y_new)

        # print(x_new,y_new)
        # input()

        return x_new, y_new



class y_update_model3(torch.nn.Module):
    
    def __init__(self,x_size,feat_size):
        super(y_update_model3,self).__init__()
        self.el = nn.ELU()
        self.feat_size = feat_size
        
    def forward(self,A,x,mu):

        eye = torch.ones(size=(A.shape[0],self.feat_size))
        # compute AX
        x = torch.matmul(A,x) - eye
        x = self.el(mu * x) + eye
        
        return x

class x_update_sub_channel(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update_sub_channel,self).__init__()
        self.feat_size = feat_size
        self.t1=torch.nn.Parameter(torch.randn(size=(feat_size, feat_size),requires_grad=True))
        self.t2=torch.nn.Parameter(torch.randn(size=(y_size, feat_size),requires_grad=True))
        self.t3=torch.nn.Parameter(torch.randn(size=(1, feat_size),requires_grad=True))
        self.act = torch.nn.Sigmoid()
        
    def forward(self,ATy,x):
        eye = torch.ones(size=ATy.shape)
        fout = torch.matmul(self.act(torch.matmul(ATy,self.t2)+self.t3) + self.act(torch.matmul(ATy,self.t2)-self.t3) - eye, self.t1) 
        return torch.mul(fout,x)

class x_update_model3(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,channels, mode='ori'):
        super(x_update_model3,self).__init__()
        self.feat_size = feat_size
        # self.channels = channels
        # self.t1 = []
        # self.t2 = []
        # self.t3 = []
        # for c in range(channels):
        #     self.t1.append(torch.nn.Parameter(torch.randn(size=(feat_size, feat_size),requires_grad=True)))
        #     self.t2.append(torch.nn.Parameter(torch.randn(size=(y_size, feat_size),requires_grad=True)))
        #     self.t3.append(torch.nn.Parameter(torch.randn(size=(1, feat_size),requires_grad=True)))
            # self.register_parameter("t1",self.t1)
            # self.register_parameter("tb1",self.tb1)
        if mode !='dchannel':
            channels+=6
        # self.xup = nn.ModuleList()
        # for indx in range(channels):
        #     self.xup.append(x_update_sub_channel(feat_size,feat_size,feat_size))
            
            
        self.xup = nn.ModuleList()
        for indx in range(channels):
            if mode == 'ori' or mode =='':
                self.xup.append(x_update_sub_channel(feat_size,feat_size,feat_size))
            elif mode =='dchannel':
                self.xup.append(x_update_dchannel_channel(feat_size,feat_size,feat_size))
                
            
        # self.t4=torch.nn.Parameter(torch.randn(size=(feat_size, feat_size),requires_grad=True))
        # self.t5=torch.nn.Parameter(torch.randn(size=(y_size, feat_size),requires_grad=True))
        # self.t6=torch.nn.Parameter(torch.randn(size=(1, feat_size),requires_grad=True))
        
        
        self.xupg = nn.ModuleList()
        for indx in range(channels):
            if mode == 'ori' or mode =='':
                self.xupg.append(x_update_sub_channel_g(feat_size,feat_size,feat_size))
            elif mode =='dchannel':
                self.xupg.append(x_update_dchannel_channel_g(feat_size,feat_size,feat_size))
        

        self.act = torch.nn.Sigmoid()
        self.rel = torch.nn.LeakyReLU()
        
        if self.feat_size > 1:
            self.out = base_emb(feat_size,feat_size)
        
    def forward(self,A,x,y):
        
        # compute AX
        ATy = torch.matmul(torch.transpose(A,0,1),y) # n x f_y
        eye = torch.ones(size=ATy.shape)
        ATy = ATy - eye

        f = None
        for index, layer in enumerate(self.xup):
            if f is None:
                f = layer(ATy, x)
            else:
                f = f + layer(ATy, x)
                
            
        # # emulate f(Ay-1) by 2 sigmoid activated mlps
        # fout = torch.matmul(self.act(torch.matmul(ATy,self.t2[0])+self.t3[0]) + self.act(torch.matmul(ATy,self.t2[0])-self.t3[0]) - eye, self.t1[0]) 
        # # print(fout)
        # # quit()
        # f = torch.mul(fout,x)
        # for i in range(1,self.channels):
        #     fout = torch.matmul(self.act(torch.matmul(ATy,self.t2[i])+self.t3[i]) + self.act(torch.matmul(ATy,self.t2[i])-self.t3[i]) - eye, self.t1[i]) 
        #     f = f + torch.mul(fout,x)


        # emulate g(Ay-1) by single sigmoid
        # g = torch.matmul(self.act(torch.matmul(ATy,self.t5)+self.t6) - self.act(self.t6), self.t4)
        g = None
        for index, layer in enumerate(self.xupg):
            if g is None:
                g = layer(ATy, x)
            else:
                g = g + layer(ATy, x)        
        
        out_res = x + f + g
        if self.feat_size > 1:
            out_res = self.out(out_res)
        return self.rel(out_res)
    
class framework_model3(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,n_layers):
        super(framework_model3,self).__init__()
        # initial channel expansion layers
        self.x_emb = base_emb(x_size,feat_size)
        self.y_emb = base_emb(y_size,feat_size)
        
        self.updates = nn.ModuleList()
        for indx in range(n_layers):
            self.updates.append(update_cycle_model3(feat_size,feat_size,feat_size))
            
        self.x_out = base_emb(feat_size,1)
        self.y_out = base_emb(feat_size,1)
        
        
    def forward(self,A,x,y,mu):
        # initial embedding
        x = self.x_emb(x)
        y = self.y_emb(y)
        for index, layer in enumerate(self.updates):
            # print(f'Starting iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')
            x,y = layer(A,x,y,mu)
            # print(f'!!!!!!Ending iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')

        x = self.x_out(x)
        y = self.y_out(y)

        return x,y
        
class framework_model1dim(torch.nn.Module):
    
    def __init__(self,n_layers,channels,nfeat = 1, mode='ori'):
        super(framework_model1dim,self).__init__()
        # initial channel expansion layers
        
        self.nfeat = nfeat
        if self.nfeat>1:
            self.x_emb = base_emb(1,nfeat,_use_bias=False)
            self.y_emb = base_emb(1,nfeat,_use_bias=False)

        self.updates = nn.ModuleList()
        for indx in range(n_layers):
            self.updates.append(update_cycle_model3(nfeat,nfeat,nfeat,channels, mode=mode))

        if self.nfeat>1:
            self.x_out = base_emb(nfeat,1,_use_bias=False)
            self.y_out = base_emb(nfeat,1,_use_bias=False)
            
        
        
    def forward(self,A,x,y,mu):
        if self.nfeat>1:
            x = self.x_emb(x)
            y = self.y_emb(y)
        # initial embedding
        for index, layer in enumerate(self.updates):
            # print(f'Starting iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')
            x,y = layer(A,x,y,mu)
            # print(f'!!!!!!Ending iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')

        if self.nfeat>1:
            x = self.x_out(x)
            y = self.y_out(y)
        return x,y

    

class framework_learn_mu(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,n_layers):
        super(framework_learn_mu,self).__init__()

        # initial channel expansion layers
        self.x_emb = base_emb(x_size,feat_size)
        self.y_emb = base_emb(y_size,feat_size)

        self.updates = nn.ModuleList()
        for indx in range(n_layers):
            self.updates.append(update_cycle(feat_size,feat_size,feat_size))
        # add the output layer:
        # self.updates.append(update_cycle(feat_size,feat_size,1))
        self.x_out = base_emb(feat_size,1)
        self.y_out = base_emb(feat_size,1)

        
    def sparse_dense_mul(self,s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())
        
    def forward(self,A,x,y,mu):
        # initial embedding
        x = self.x_emb(x)
        y = self.y_emb(y)


        for index, layer in enumerate(self.updates):
            # print(f'Starting iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')
            x,y = layer(A,x,y,mu)
            # print(f'!!!!!!Ending iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')

        x = self.x_out(x)
        y = self.y_out(y)

        return x,y



class framework_fixed_mu(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,n_layers,):
        super(framework_fixed_mu,self).__init__()

        # initial channel expansion layers
        self.x_emb = base_emb(x_size,feat_size)
        self.y_emb = base_emb(y_size,feat_size)

        self.updates = nn.ModuleList()
        for indx in range(n_layers):
            self.updates.append(update_cycle_with_mu(feat_size,feat_size,feat_size))
        # add the output layer:
        # self.updates.append(update_cycle(feat_size,feat_size,1))
        self.x_out = base_emb(feat_size,1)
        self.y_out = base_emb(feat_size,1)

        
    def sparse_dense_mul(self,s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())
        
    def forward(self,A,x,y,mu):
        # initial embedding
        x = self.x_emb(x)
        y = self.y_emb(y)


        for index, layer in enumerate(self.updates):
            # print(f'Starting iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')
            x,y = layer(A,x,y,mu)
            # print(f'!!!!!!Ending iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')

        x = self.x_out(x)
        y = self.y_out(y)

        return x,y


class update_cycle(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size):
        super(update_cycle,self).__init__()
        self.y_up = y_update(x_size,feat_size)
        self.x_up = x_update(x_size, feat_size,feat_size)
        
    def forward(self,A,x,y):

        # compute AX
        y_new = self.y_up(A,x)
        x_new = self.x_up(A,x,y_new)

        return x_new, y_new




class y_update(torch.nn.Module):
    
    def __init__(self,x_size,feat_size):
        super(y_update,self).__init__()
        self.mlp = nn.Sequential(
            # AX ->W  +  ->b
            nn.Linear(x_size,feat_size,bias=True),
            nn.ELU(),
        )
        
    def forward(self,A,x):

        # compute AX
        x = torch.matmul(A,x)
        return self.mlp(x)

class x_update(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update,self).__init__()
        self.mlp_x = nn.Sequential(
            # x + f(Ay - 1) x + g(Ay-1)
            nn.Linear(x_size,feat_size,bias=True),
            nn.LeakyReLU()
        )

        self.mlp_f1 = nn.Sequential(
            # x + f(Ay - 1) x + g(Ay-1)
            nn.Linear(y_size,feat_size,bias=True),
            nn.Sigmoid()
        )
        self.mlp_f2 = nn.Sequential(
            # x + f(Ay - 1) x + g(Ay-1)
            nn.Linear(y_size,feat_size,bias=True),
            nn.Sigmoid()
        )
        
        self.mlp_g = nn.Sequential(
            # x + f(Ay - 1) x + g(Ay-1)
            nn.Linear(y_size,feat_size,bias=True),
            nn.Sigmoid()
        )
        
    def forward(self,A,x,y):
        
        x = self.mlp_x(x)

        # compute AX
        ATy = torch.matmul(torch.transpose(A,0,1),y) # n x f_y


        # emulate f(Ay-1) by 2 sigmoid activated mlps
        y1 = self.mlp_f1(ATy)  # n x f_y1  
        y2 = self.mlp_f2(ATy)  # n x f_y2
        # print(f'ATy size:{ATy.shape}  y1 size:{y1.shape}, y2 size:{y2.shape}')
        # elementwise multiplication by x 
        f = torch.mul((y1+y2),x)

        # emulate g(Ay-1) by single sigmoid
        g = self.mlp_g(ATy)
    
        return x + f + g


class base_emb(torch.nn.Module):
    
    def __init__(self,in_size,out_size,_use_bias=True):
        super(base_emb,self).__init__()
        # print(in_size,"->",out_size)
        self.emb=nn.Sequential(
            nn.Linear(in_size,out_size,bias=_use_bias),
            nn.LeakyReLU(),
        )
        
    def forward(self,x):
        # print(x.shape)
        return self.emb(x)


class update_cycle_with_mu(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size):
        super(update_cycle_with_mu,self).__init__()
        self.y_up = y_update_with_mu(x_size,feat_size)
        self.x_up = x_update(x_size, feat_size,feat_size)
        
    def forward(self,A,x,y,mu):

        # compute AX
        y_new = self.y_up(A,x,mu)
        x_new = self.x_up(A,x,y_new)
        return x_new, y_new


class y_update_with_mu(torch.nn.Module):
    
    def __init__(self,x_size,feat_size):
        super(y_update_with_mu,self).__init__()
        self.feat_size = feat_size
        self.mlp = nn.Sequential(
            # AX ->W  +  ->b
            nn.Linear(x_size,feat_size,bias=False),
            nn.LeakyReLU(),
        )
        def init_w(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(1e-20)

        self.mlp.apply(init_w)
        
    def forward(self,A,x,mu):

        # compute AX
        eye = torch.ones(size=(A.shape[0],self.feat_size))
        x = torch.matmul(A,x)
        # print(x)
        # print(x.shape,eye.shape)
        x = self.mlp(x)-eye
        x = mu*x
        x = torch.exp(x)
        # print(x)
        # input()
        return x
    
    
    
    

class GCN_layer(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size):
        super(GCN_layer,self).__init__()
        self.feat_size = feat_size
        self.embx = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.emby = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.outlayer = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        
        self.embx2 = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.emby2 = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        self.outlayer2 = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,feat_size,bias=True),
        )
        
        # def _initialize_weights(m):
        #     for m in self.modules():
        #         if isinstance(m,nn.Linear):
        #             torch.nn.init.xavier_uniform_(m.weight,gain=1)
                    
        # self.embx.apply(_initialize_weights)
        # self.emby.apply(_initialize_weights)
        # self.embx2.apply(_initialize_weights)
        # self.emby2.apply(_initialize_weights)
        # self.outlayer.apply(_initialize_weights)
        # self.outlayer2.apply(_initialize_weights)
        
    def forward(self,A,x,y):

        Ax = self.embx(torch.matmul(A,x))
        y = self.emby(y) + Ax
        y = self.outlayer(y)
        AT = torch.transpose(A,0,1)
        ATy = self.emby2(torch.matmul(AT,y))
        x = self.embx2(x) + ATy
        x = self.outlayer2(x)
        return x,y
    


class GCN(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,n_layer=4):
        super(GCN,self).__init__()
        self.embx = nn.Sequential(
            nn.Linear(x_size,feat_size,bias=True),
        )
        self.emby = nn.Sequential(
            nn.Linear(y_size,feat_size,bias=True),
        )
        self.updates = nn.ModuleList()
        self.updates.append(GCN_layer(feat_size,feat_size,feat_size))
        for indx in range(n_layer):
            self.updates.append(GCN_layer(feat_size,feat_size,feat_size))
            
        self.outlayer = nn.Sequential(
            nn.Linear(feat_size,feat_size,bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size,1,bias=True),
            nn.LeakyReLU(),
        )
        
        # def _initialize_weights(m):
        #     for m in self.modules():
        #         if isinstance(m,nn.Linear):
        #             torch.nn.init.xavier_uniform_(m.weight,gain=1)
                    
        # self.outlayer.apply(_initialize_weights)
        # self.embx.apply(_initialize_weights)
        # self.emby.apply(_initialize_weights)
        
        
    def forward(self,A,x,y):
        x = self.embx(x)
        y = self.embx(y)

        for index, layer in enumerate(self.updates):
            x,y = layer(A,x,y)
            
        return self.outlayer(x)
        
            





class framework_model1dim_covering(torch.nn.Module):
    
    def __init__(self,n_layers,channels,nfeat = 1, mode='ori'):
        super(framework_model1dim_covering,self).__init__()
        # initial channel expansion layers
        
        self.nfeat = nfeat
        if self.nfeat>1:
            self.x_emb = base_emb(1,nfeat,_use_bias=False)
            self.y_emb = base_emb(1,nfeat,_use_bias=False)
            # print('x emb:',count_parameters(self.x_emb))
            # print('y emb:',count_parameters(self.y_emb))

        self.updates = nn.ModuleList()
        for indx in range(n_layers):
            self.updates.append(update_cycle_model3_covering(nfeat,nfeat,nfeat,channels,mode=mode))
        # for indx,m in enumerate(self.updates):
        #     print(indx,count_parameters(m))
        # print(count_parameters(self.y_up))

        if self.nfeat>1:
            self.x_out = base_emb(nfeat,1,_use_bias=False)
            self.y_out = base_emb(nfeat,1,_use_bias=False)
            # print('x emb:',count_parameters(self.x_out))
            # print('y emb:',count_parameters(self.y_out))
            
        
        
    def forward(self,A,x,y,mu):
        if self.nfeat>1:
            x = self.x_emb(x)
            y = self.y_emb(y)
        # initial embedding
        for index, layer in enumerate(self.updates):
            # print(f'Starting iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')
            x,y = layer(A,x,y,mu)
            # print(f'!!!!!!Ending iteration{index}: A:{A.shape}, x:{x.shape}, y:{y.shape}')

        if self.nfeat>1:
            x = self.x_out(x)
            y = self.y_out(y)
        return x,y




class update_cycle_model3_covering(torch.nn.Module):
    
    def __init__(self,x_size,y_size,feat_size,channels=16, mode='ori'):
        super(update_cycle_model3_covering,self).__init__()
        self.y_up = y_update_model3_covering(x_size,feat_size)
        # for indx in range(n_layers):
        #     self.updates.append(x_update_model3(x_size, feat_size,feat_size))
        self.x_up = x_update_model3_covering(x_size, feat_size,feat_size,channels, mode=mode)
        
        self.feat_size = feat_size
        # if self.feat_size > 1:
        #     self.out = base_emb(feat_size,feat_size)
        
    def forward(self,A,x,y,mu):

        # compute AX
        y_new = self.y_up(A,x,mu)
        x_new = self.x_up(A,x,y_new)

        # if self.feat_size > 1:
        #     x_new = self.out(x_new)
        # print(x_new,y_new)
        # input()

        return x_new, y_new



class y_update_model3_covering(torch.nn.Module):
    
    def __init__(self,x_size,feat_size):
        super(y_update_model3_covering,self).__init__()
        self.el = nn.ELU()
        self.feat_size = feat_size
        
    def forward(self,A,x,mu):

        eye = torch.ones(size=(A.shape[0],self.feat_size))
        # compute AX
        x =  eye - torch.matmul(A,x)
        x = self.el(mu * x) + eye
        
        return x

class x_update_dchannel_channel(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update_dchannel_channel,self).__init__()
        self.feat_size = feat_size
        self.t1=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t2=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t3=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.act = torch.nn.Sigmoid()
        
    def forward(self,ATy,x):
        eye = torch.ones(size=ATy.shape)
        fout = torch.mul(self.act(ATy*self.t2+self.t3) + self.act(ATy*self.t2-self.t3) - eye,self.t1)
        return torch.mul(fout,x)


class x_update_dchannel_channel_g(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update_dchannel_channel_g,self).__init__()
        self.feat_size = feat_size
        self.t1=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t2=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t3=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.act = torch.nn.Sigmoid()
        
    def forward(self,ATy,x):
        eye = torch.ones(size=ATy.shape)
        fout = torch.mul(self.act(ATy*self.t2+self.t3) + self.act(ATy*self.t2-self.t3) - eye,self.t1)
        return torch.mul(fout,x)

class x_update_sub_channel_g(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update_sub_channel_g,self).__init__()
        self.feat_size = feat_size
        self.t4=torch.nn.Parameter(torch.randn(size=(feat_size, feat_size),requires_grad=True))
        self.t5=torch.nn.Parameter(torch.randn(size=(y_size, feat_size),requires_grad=True))
        self.t6=torch.nn.Parameter(torch.randn(size=(1, feat_size),requires_grad=True))
        self.act = torch.nn.Sigmoid()
        
    def forward(self,ATy,x):
        eye = torch.ones(size=ATy.shape)
        g = torch.matmul(self.act(torch.matmul(ATy,self.t5)+self.t6) - self.act(self.t6), self.t4)
        return g
    
    
class x_update_dchannel_channel_g(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size):
        super(x_update_dchannel_channel_g,self).__init__()
        self.feat_size = feat_size
        self.t4=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t5=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.t6=torch.nn.Parameter(torch.randn(size=(1, 1),requires_grad=True))
        self.act = torch.nn.Sigmoid()
        
    def forward(self,ATy,x):
        eye = torch.ones(size=ATy.shape)
        g = (self.act(ATy*self.t5+self.t6) - self.act(self.t6))*self.t4
        return g


class x_update_model3_covering(torch.nn.Module):
    
    def __init__(self,x_size, y_size, feat_size,channels, mode='ori'):
        super(x_update_model3_covering,self).__init__()
        self.feat_size = feat_size
        if mode !='dchannel':
            channels+=6
        self.xup = nn.ModuleList()
        for indx in range(channels):
            if mode == 'ori' or mode =='':
                self.xup.append(x_update_sub_channel(feat_size,feat_size,feat_size))
            elif mode =='dchannel':
                self.xup.append(x_update_dchannel_channel(feat_size,feat_size,feat_size))
                

        self.xupg = nn.ModuleList()
        for indx in range(channels):
            if mode == 'ori' or mode =='':
                self.xupg.append(x_update_sub_channel_g(feat_size,feat_size,feat_size))
            elif mode =='dchannel':
                self.xupg.append(x_update_dchannel_channel_g(feat_size,feat_size,feat_size))
        
        self.rel = torch.nn.LeakyReLU()

        self.act = torch.nn.Sigmoid()
        if self.feat_size > 1:
            self.out = base_emb(feat_size,feat_size,_use_bias=False)
        

        
    def forward(self,A,x,y):
        
        # compute AX
        ATy = torch.matmul(torch.transpose(A,0,1),y) # n x f_y
        eye = torch.ones(size=ATy.shape)
        ATy = eye - ATy  

        f = None
        for index, layer in enumerate(self.xup):
            if f is None:
                f = layer(ATy, x)
            else:
                f = f + layer(ATy, x)

        
        g = None
        for index, layer in enumerate(self.xupg):
            if g is None:
                g = layer(ATy, x)
            else:
                g = g + layer(ATy, x)        

        outr = x + f + g
        # return torch.clamp(x + f + g,min=0.0,max=1.0)
        if self.feat_size > 1:
            outr = self.out(outr)
        return self.rel(outr)
    
        # return x + f + g