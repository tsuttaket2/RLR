#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('PS')  # generate postscript output by default

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
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default='.')
parser.add_argument('--listfile_folder', type=str, help='listfile_folder', default='./list folder/')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='1')
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=10)
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--timestep', type=float, default=1.0)
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
parser.add_argument('--dropout_p', type=float, default=0., help='dropout_p')
parser.add_argument('--output_dropout_p', type=float, default=0., help='output_dropout_p')
parser.add_argument('--emb_dim', type=int, default=57221, help='embedding_dim')
parser.add_argument('--visit_dim', type=int, default=96, help='visit_dim')
parser.add_argument('--var_dim', type=int, default=96, help='var_dim')

parser.add_argument('--train_nbatches', type=int, default=30, help='train_nbatches')
parser.add_argument('--valid_nbatches', type=int, default=20, help='valid_nbatches')
parser.set_defaults(small_part=False)

args = parser.parse_args()
print(args)
""" Helper Functions """


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


""" Custom Dataset """


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features, reverse=True):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
			reverse (bool): If true, reverse the order of sequence (for RETAIN)
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.seqs = []
		# self.labels = []

		for seq, label in zip(seqs, labels):

			if reverse:
				sequence = list(reversed(seq))
			else:
				sequence = seq

			row = []
			col = []
			val = []
			for i, visit in enumerate(sequence):
				for code in visit:
					if code < num_features:
						row.append(i)
						col.append(code)
						val.append(1.0)

			self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
										shape=(len(sequence), num_features)))
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""


# @profile
def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor
	:returns
		seqs
		labels
		lengths
	"""
	batch_seq, batch_label = zip(*batch)

	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_padded_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]

		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()

		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	seq_tensor = np.stack(sorted_padded_seqs, axis=0)
	label_tensor = torch.LongTensor(sorted_labels)

	return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths)


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
args.timestep=1.0
from Thiti_Model import my_utils
from sklearn.metrics import log_loss



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)







train_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                    listfile=os.path.join(args.listfile_folder, 'train_listfile.csv'))
valid_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'valid_listfile.csv'))

# Set number of batches in one epoch
train_nbatches = args.train_nbatches
valid_nbatches = args.valid_nbatches

train_data_gen = BatchGen(train_reader, args.batch_size, train_nbatches, True,return_names=True)
valid_data_gen = BatchGen(valid_reader, args.batch_size, valid_nbatches, False,return_names=True)

args.input_dim=train_reader.read_example(0)['X'].shape[1]
# In[4]:


model = RETAIN(dim_input=args.input_dim,
                   dim_emb=args.emb_dim,
                   dropout_emb=args.dropout_p,
                   dim_alpha=args.visit_dim,
                   dim_beta=args.var_dim,
                   dropout_context=args.output_dropout_p,
                   dim_output=2).to(device)
n_params=count_parameters(model)
print("#parameters ",n_params)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()


max_auprc = 0
file_name = './saved_weights/'+args.file_name
for each_chunk in range(args.epochs):
    model.train()
    cur_batch_loss = []
    for each_batch in range(train_data_gen.steps):
        batch_data = next(train_data_gen)
        
        batch_y = batch_data['data'][1]
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
        batch_masks = batch_data['masks']
        batch_masks = torch.tensor(batch_masks, dtype=torch.float32).to(device)
        
        batch_data=batch_data['data'][0]
        batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)
            
        batch_masks=torch.sum(torch.tensor(batch_masks),dim=1)
        optimizer.zero_grad()
        masked_output,_,_=model(batch_data,batch_masks)
        masked_output=F.softmax(masked_output)[:,1]
        
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
        val_loss=[]
        valid_true = []
        valid_pred = []
        for each_batch in range(valid_data_gen.steps):
            batch_data = next(valid_data_gen)
            batch_y = batch_data['data'][1]
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_masks = batch_data['masks']
            batch_masks = torch.tensor(batch_masks, dtype=torch.float32).to(device)

            batch_data=batch_data['data'][0]
            batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)

            batch_masks=torch.sum(torch.tensor(batch_masks),dim=1)
            optimizer.zero_grad()
            masked_output,_,_=model(batch_data,batch_masks)
            masked_output=F.softmax(masked_output)[:,1]
            
            loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
            loss = torch.mean(loss)
            loss = torch.neg(torch.sum(loss))
            cur_val_loss.append(loss.cpu().detach().numpy())
            
            for t, p in zip(batch_y.cpu().numpy().flatten(), masked_output.cpu().detach().numpy().flatten()):
                valid_true.append(t)
                valid_pred.append(p)
        val_loss.append(np.mean(np.array(cur_val_loss)))
        print('Valid loss = %.4f'%(val_loss[-1]))
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


# In[11]:


try:
    del val_data_gen
    del train_data_gen
    del train_reader
    del val_reader
except:
    pass
'''Evaluate phase'''
print('Testing model ... ')

checkpoint = torch.load(file_name)
save_chunk = checkpoint['chunk']
print("last saved model is in chunk {}".format(save_chunk))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()

test_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'test_listfile.csv'))
test_nbatches=None
test_data_gen = BatchGen_Fulldata(test_reader, args.batch_size, test_nbatches, False,return_names=True)

with torch.no_grad():
    cur_test_loss = []
    test_true = []
    test_pred = []
    for each_batch in range(test_data_gen.steps):
        batch_data = next(test_data_gen)
        
        batch_y = batch_data['data'][1]
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
        batch_masks = batch_data['masks']
        batch_masks = torch.tensor(batch_masks, dtype=torch.float32).to(device)

        batch_data=batch_data['data'][0]
        batch_data=torch.tensor(batch_data, dtype=torch.float32).to(device)

        batch_masks=torch.sum(torch.tensor(batch_masks),dim=1)
        masked_output,_,_=model(batch_data,batch_masks)
        masked_output=F.softmax(masked_output)[:,1]

        loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
        loss = torch.mean(loss)
        loss = torch.neg(torch.sum(loss))
        cur_test_loss.append(loss.cpu().detach().numpy())

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




