#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse


# In[13]:


parser = argparse.ArgumentParser()

parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--r_visit', type=int, default=4, help='Compress ration r for visit features')
parser.add_argument('--r_conv', type=int, default=4, help='Compress ration r for convolutional features')
parser.add_argument('--kernel_size', type=int, default=2, help='Convolutional kernel size')
parser.add_argument('--kernel_num', type=int, default=64, help='Number of convolutional filters')
parser.add_argument('--activation_func', type=str, default='sigmoid', help='Activation function for feature recalibration (sigmoid / sparsemax)')

parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                default='/home/thitis/Research/Research_SGH/data/mimic3benchmark/data/in-hospital-mortality')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='1')
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=100)
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--timestep', type=float, default=1.0)
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=0.00, help='clip')
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
args = parser.parse_args()


# In[16]:


print(args)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import numpy as np
import os
import imp
import re

from mimic3models import common_utils
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from mimic3models import metrics
device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')

args.timestep=1.0
from Thiti_Model import my_utils
from Thiti_Model.Adacare_model import  AdaCare

from sklearn.metrics import log_loss
# In[15]:


train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
normalizer_state = os.path.join(args.data, normalizer_state)
normalizer.load_params(normalizer_state)

# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

# Generators
training_set = my_utils.Dataset(train_raw)
training_generator = torch.utils.data.DataLoader(training_set,batch_size = args.batch_size, shuffle= True)

validation_set = my_utils.Dataset(val_raw)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = args.batch_size, shuffle= False)


# Load test data and prepare generators
test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                       listfile=os.path.join(args.data, 'test_listfile.csv'),
                                       period_length=48.0)
test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)
test_set = my_utils.Dataset(test_raw)
test_generator = torch.utils.data.DataLoader(test_set,batch_size = args.batch_size, shuffle= False)


# In[26]:


model = AdaCare(args.rnn_dim, args.kernel_size, args.kernel_num, args.input_dim, args.output_dim, args.dropout_rate, args.r_visit, args.r_conv, args.activation_func, device).to(device)


# In[28]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
n_params=count_parameters(model)
print("#parameters ",n_params)


# In[20]:


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip>0:
    my_utils.enable_gradient_clipping(model, clip=args.clip)



print('Start training ... ')

train_loss = []
val_loss = []
batch_loss = []
max_auprc = 0


file_name = './saved_weights/'+args.file_name
for each_chunk in range(args.epochs):
    cur_batch_loss = []
    model.train()
    each_batch=0
    
    for x_batch, y_batch in training_generator:
        optimizer.zero_grad()
        cur_output, _ = model(x_batch, device)
        cur_output = cur_output.squeeze(-1)
        cur_output = cur_output[:,-1]
        loss = y_batch * torch.log(cur_output + 1e-7) + (1 - y_batch) * torch.log(1 - cur_output + 1e-7)
        loss = torch.mean(loss)
        loss = torch.neg(torch.sum(loss))
        cur_batch_loss.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        each_batch=each_batch+1        
        if each_batch % 50 == 0:
            print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))
    batch_loss.append(cur_batch_loss)
    train_loss.append(np.mean(np.array(cur_batch_loss)))
    
    print("\n==>Predicting on validation")
    with torch.no_grad():
        model.eval()
        cur_val_loss = []
        valid_true = []
        valid_pred = []
        each_batch=0
        for valid_x,valid_y in validation_generator:
            valid_output,_ = model(valid_x,device)
            valid_output = valid_output.squeeze(-1)
            valid_output = valid_output[:,-1]
        
            valid_loss = valid_y * torch.log(valid_output + 1e-7) + (1 - valid_y) * torch.log(1 - valid_output + 1e-7)
            valid_loss = torch.mean(valid_loss)
            valid_loss = torch.neg(torch.sum(valid_loss))
            cur_val_loss.append(valid_loss.cpu().detach().numpy())
            
            for t, p in zip(valid_y.cpu().numpy().flatten(), valid_output.cpu().detach().numpy().flatten()):
                valid_true.append(t)
                valid_pred.append(p)
        
        val_loss.append(np.mean(np.array(cur_val_loss)))
        print('Valid loss = %.4f'%(val_loss[-1]))
        print('\n')
        valid_pred = np.array(valid_pred)
        valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
        ret = metrics.print_metrics_binary(valid_true, valid_pred)
        print()
        
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


# In[45]:


# Load test data and prepare generators
test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                       listfile=os.path.join(args.data, 'test_listfile.csv'),
                                       period_length=48.0)
test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part)
test_set = my_utils.Dataset(test_raw)
test_generator = torch.utils.data.DataLoader(test_set,batch_size = args.batch_size, shuffle= False)

'''Evaluate phase'''
print('Testing model ... ')

checkpoint = torch.load(file_name)
save_chunk = checkpoint['chunk']
print("last saved model is in chunk {}".format(save_chunk))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


with torch.no_grad():
    cur_test_loss = []
    test_true = []
    test_pred = []
    
    for test_x,test_y in test_generator:
        test_output,_ = model(test_x,device)
        test_output = test_output.squeeze(-1)
        test_output = test_output[:,-1]
            
        test_loss = test_y * torch.log(test_output + 1e-7) + (1 - test_y) * torch.log(1 - test_output + 1e-7)
        test_loss = torch.mean(test_loss)
        test_loss = torch.neg(torch.sum(test_loss))
        cur_test_loss.append(test_loss.cpu().detach().numpy())
        
        
        for t, p in zip(test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
            test_true.append(t)
            test_pred.append(p)
    
    print('Test neg log-likelihood = %.4f'%(log_loss(test_true, test_pred)))
    print('\n')
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    test_ret = metrics.print_metrics_binary(test_true, test_pred)


# In[ ]:




