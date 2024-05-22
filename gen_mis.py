from helper import *
import os

if not os.path.isdir('data_mis'):
    os.mkdir('data_mis')
if not os.path.isdir('data_mis/test'):
    os.mkdir('data_mis/test')
if not os.path.isdir('data_mis/valid'):
    os.mkdir('data_mis/valid')
if not os.path.isdir('data_mis/train'):
    os.mkdir('data_mis/train')

generate_dateset_MIS(train_files=1000,valid_files=100,test_files=100)
