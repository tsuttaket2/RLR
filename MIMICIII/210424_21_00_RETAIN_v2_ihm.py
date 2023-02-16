#!/usr/bin/env python
# coding: utf-8

# In[15]:


import argparse
from collections import OrderedDict
import importlib

parser = argparse.ArgumentParser()
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
parser.add_argument('--dropout_emb', type=float, default=0., help='dropout_p')
parser.add_argument('--dropout_context', type=float, default=0., help='output_dropout_p')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding_dim')
parser.add_argument('--dim_alpha', type=int, default=128, help='visit_dim')
parser.add_argument('--dim_beta', type=int, default=128, help='var_dim')
parser.add_argument('--num_features', type=int, default=76, help='num_features')
parser.set_defaults(small_part=False)


args = parser.parse_args()
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
from sklearn.metrics import log_loss

# In[ ]:


# Build readers, discretizers, normalizers
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


# In[10]:


""" RETAIN model class """


class RETAIN(nn.Module):
	def __init__(self, dim_input, dim_emb=128, dropout_input=0.8, dropout_emb=0.5, dim_alpha=128, dim_beta=128,
				 dropout_context=0.5, dim_output=2, l2=0.0001, batch_first=True):
		super(RETAIN, self).__init__()
		self.batch_first = batch_first
		self.embedding = nn.Sequential(
			nn.Dropout(p=dropout_input),
			nn.Linear(dim_input, dim_emb, bias=False),
			nn.Dropout(p=dropout_emb)
		)
		init.xavier_normal(self.embedding[1].weight)

		self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

		self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
		init.xavier_normal(self.alpha_fc.weight)
		self.alpha_fc.bias.data.zero_()

		self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

		self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
		init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
		self.beta_fc.bias.data.zero_()

		self.output = nn.Sequential(
			nn.Dropout(p=dropout_context),
			nn.Linear(in_features=dim_emb, out_features=dim_output)
		)
		init.xavier_normal(self.output[1].weight)
		self.output[1].bias.data.zero_()

	def forward(self, x, lengths):
		if self.batch_first:
			batch_size, max_len = x.size()[:2]
		else:
			max_len, batch_size = x.size()[:2]

		# emb -> batch_size X max_len X dim_emb
		emb = self.embedding(x)

		packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first,enforce_sorted=False)

		g, _ = self.rnn_alpha(packed_input)

		# alpha_unpacked -> batch_size X max_len X dim_alpha
		alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

		# mask -> batch_size X max_len X 1
		mask = Variable(torch.FloatTensor(
			[[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
						requires_grad=False)
		if next(self.parameters()).is_cuda:  # returns a boolean
			mask = mask.cuda()

		# e => batch_size X max_len X 1
		e = self.alpha_fc(alpha_unpacked)

		def masked_softmax(batch_tensor, mask):
			exp = torch.exp(batch_tensor)
			masked_exp = exp * mask
			sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
			return masked_exp / sum_masked_exp

		# Alpha = batch_size X max_len X 1
		# alpha value for padded visits (zero) will be zero
		alpha = masked_softmax(e, mask)

		h, _ = self.rnn_beta(packed_input)

		# beta_unpacked -> batch_size X max_len X dim_beta
		beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

		# Beta -> batch_size X max_len X dim_emb
		# beta for padded visits will be zero-vectors
		beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

		# context -> batch_size X (1) X dim_emb (squeezed)
		# Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
		# Vectorized sum
		context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

		# without applying non-linearity
		logit = self.output(context)

		return logit, alpha, beta


# In[12]:





# In[5]:


model = RETAIN(dim_input=args.num_features,
				   dim_emb=args.emb_dim,
				   dropout_emb=args.dropout_emb,
				   dim_alpha=args.dim_alpha,
				   dim_beta=args.dim_beta,
				   dropout_context=args.dropout_context,
				   dim_output=2).to(device)


# In[6]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
n_params=count_parameters(model)
print("#parameters ",n_params)


# In[16]:


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip>0:
    my_utils.enable_gradient_clipping(model, clip=args.clip)


# In[46]:


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
        mask=torch.logical_not(torch.sum(x_batch==0,dim=2) == x_batch.shape[2]).to(device)
        mask=torch.sum(mask,dim=1)
        optimizer.zero_grad()
        cur_output,_,_=model(x_batch,mask)
        cur_output=F.softmax(cur_output)[:,1]
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
            valid_mask=torch.logical_not(torch.sum(valid_x==0,dim=2) == valid_x.shape[2]).to(device)
            valid_mask=torch.sum(valid_mask,dim=1)
            
            valid_output,_,_ = model(valid_x,valid_mask)
            valid_output = F.softmax(valid_output)[:,1]
            
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


# In[47]:


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
        test_mask=torch.logical_not(torch.sum(test_x==0,dim=2) == test_x.shape[2]).to(device)
        test_mask=torch.sum(test_mask,dim=1)
            
        test_output,_,_ = model(test_x,test_mask)
        test_output = F.softmax(test_output)[:,1]
        
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




