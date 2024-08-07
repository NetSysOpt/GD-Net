from helper import *

tfs = 0
vfs = 0
tss = 100
generate_dateset_Maxflow(n=600,m=600,p=0.6,train_files=tfs,valid_files=vfs,test_files=tss)
generate_dateset_Maxflow(n=1000,m=1000,p=0.6,train_files=tfs,valid_files=vfs,test_files=tss)
# generate_dateset_Maxflow(n=8000,m=8000,p=1e-5,train_files=tfs,valid_files=vfs,test_files=tss)
# generate_dateset_Maxflow(n=2000,m=2000,p=0.5,train_files=tfs,valid_files=vfs,test_files=tss,mode=0)
