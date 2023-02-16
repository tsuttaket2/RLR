#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib

""" Imports """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys
import argparse
import pickle
import time
import random
import math

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of sgh task',
                    default='.')
parser.add_argument('--listfile_folder', type=str, help='folder containing listfile',
                    default='.')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='1')
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=10)
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
parser.add_argument('--patient', type=int, default=10,
                        help='patient')
parser.add_argument('--input_dim', type=int, default=57221, help='Dimension of visit record data')
parser.add_argument('--emb_dim', type=int, default=700, help='Dimension of embedding')
parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--r_visit', type=int, default=4, help='Compress ration r for visit features')
parser.add_argument('--r_conv', type=int, default=4, help='Compress ration r for convolutional features')
parser.add_argument('--kernel_size', type=int, default=2, help='Convolutional kernel size')
parser.add_argument('--kernel_num', type=int, default=64, help='Number of convolutional filters')
parser.add_argument('--activation_func', type=str, default='sigmoid', help='Activation function for feature recalibration (sigmoid / sparsemax)')


parser.add_argument('--train_nbatches', type=int, default=20, help='train_nbatches')
parser.add_argument('--valid_nbatches', type=int, default=20, help='val_nbatches')
parser.set_defaults(small_part=False)

args = parser.parse_args()
print(args)


# In[2]:



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import numpy as np
import os
import imp
import re

from resources.utils.data_reader import SGHReader
from resources.utils.preprocessing import Discretizer
from resources.utils.mortality_generator import BatchGen,BatchGen_Fulldata
import torch
import numpy as np
from mimic3models import metrics

device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')
from Thiti_Model import my_utils
from Thiti_Model.Adacare_emb_model import AdaCare

from sklearn.metrics import log_loss
# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                    listfile=os.path.join(args.listfile_folder, 'train_listfile.csv'))
valid_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'valid_listfile.csv'))


train_nbatches=args.train_nbatches
valid_nbatches=args.valid_nbatches

train_data_gen = BatchGen(train_reader, args.batch_size, train_nbatches, True,return_names=True)
valid_data_gen = BatchGen(valid_reader, args.batch_size, valid_nbatches, False,return_names=True)


# In[4]:


args.input_dim=train_reader.read_example(0)['X'].shape[1]
model = AdaCare(args.rnn_dim, 
                args.kernel_size, 
                args.kernel_num, 
                args.input_dim, 
                args.output_dim, 
                args.dropout_rate, 
                args.r_visit, 
                args.r_conv, 
                args.activation_func, 
                args.emb_dim,
                device=device).to(device)


# In[5]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
n_params=count_parameters(model)
print("#parameters ",n_params)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# In[7]:


max_auprc = 0
file_name = './saved_weights/'+args.file_name
for each_chunk in range(args.epochs):
    model.train()
    cur_batch_loss = []
    for each_batch in range(train_data_gen.steps):
        batch_data = next(train_data_gen)
        
        batch_y = batch_data['data'][1]
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
        batch_mask_T = batch_data['masks_T']
        batch_mask_T = torch.tensor(batch_mask_T, dtype=torch.float32).to(device)
        
        batch_data=batch_data['data'][0]
        batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)
            
        
        optimizer.zero_grad()
        masked_output,_ = model(batch_data,device)
        masked_output = masked_output.squeeze(-1)
        masked_output = masked_output[batch_mask_T.nonzero(as_tuple=True)]
        
        loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
        loss = torch.mean(loss)
        loss = torch.neg(torch.sum(loss))
        cur_batch_loss.append(loss.cpu().detach().numpy())
        
        loss.backward()
        optimizer.step()
        
        if each_batch % 10 == 0:
            print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))
            
    print("\n==>Predicting on validation")
    del batch_data
    with torch.no_grad():
        model.eval()
        cur_val_loss = []
        valid_true = []
        valid_pred = []
        for each_batch in range(valid_data_gen.steps):
            batch_data = next(valid_data_gen)
            batch_y = batch_data['data'][1]
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_mask_T = batch_data['masks_T']
            batch_mask_T = torch.tensor(batch_mask_T, dtype=torch.float32).to(device)

            batch_data=batch_data['data'][0]
            batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)

            masked_output,_ = model(batch_data,device)
            masked_output = masked_output.squeeze(-1)
            masked_output = masked_output[batch_mask_T.nonzero(as_tuple=True)]
            
            for t, p in zip(batch_y.cpu().numpy().flatten(), masked_output.cpu().detach().numpy().flatten()):
                valid_true.append(t)
                valid_pred.append(p)
        print('\n')
        valid_pred = np.array(valid_pred)
        valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
        try:
            ret = metrics.print_metrics_binary(valid_true, valid_pred)
            print()
        except:
            continue
        cur_auprc = ret['auprc']
        if cur_auprc > max_auprc:
            max_auprc = cur_auprc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'chunk': each_chunk
                }
            torch.save(state, file_name)
            print('\n------------ Save best model ------------\n')


# In[8]:



try:
    del valid_data_gen
    del train_data_gen
    del train_reader
    del valid_reader
except:
    pass

test_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'test_listfile.csv'))

test_nbatches=None
test_data_gen = BatchGen_Fulldata(test_reader, args.batch_size, test_nbatches, False,return_names=True)

'''Evaluate phase'''
print('Testing model ... ')

checkpoint = torch.load(file_name)
save_chunk = checkpoint['chunk']
print("last saved model is in chunk {}".format(save_chunk))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


# In[11]:


with torch.no_grad():
    cur_test_loss = []
    test_true = []
    test_pred = []
    for each_batch in range(test_data_gen.steps):
        batch_data = next(test_data_gen)
        
        batch_y = batch_data['data'][1]
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
        batch_mask_T = batch_data['masks_T']
        batch_mask_T = torch.tensor(batch_mask_T, dtype=torch.float32).to(device)

        batch_data=batch_data['data'][0]
        batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)

        masked_output,_ = model(batch_data,device)
        masked_output = masked_output.squeeze(-1)
        masked_output = masked_output[batch_mask_T.nonzero(as_tuple=True)]

        for t, p in zip(batch_y.cpu().numpy().flatten(), masked_output.cpu().detach().numpy().flatten()):
            test_true.append(t)
            test_pred.append(p)
        
        if each_batch % 100 == 0:
            print('Batch %d from %d'%(each_batch, test_data_gen.steps))
    
    print('Test neg log-likelihood = %.4f'%(log_loss(test_true, test_pred)))   
    print('\n')
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    test_ret = metrics.print_metrics_binary(test_true, test_pred)


# In[ ]:




