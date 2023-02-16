import argparse
from collections import OrderedDict
from Thiti_Model.my_utils import add_common_arguments,str2bool

parser = argparse.ArgumentParser()
add_common_arguments(parser)
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default='/home/thiti/Research_SGH/data/temp/mimic3benchmark/decompensation')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--pattern_specs', type=str, help='pattern specs',
                    default='5-5_20-5_40-5')
parser.add_argument('--mlp_hidden_dim', type=int, help='mlp_hidden_dim', default='128')
parser.add_argument('--num_mlp_layers', type=int, help='num_mlp_layers', default='1')
parser.add_argument('--input_dim', type=int, help='input dimension', default='76')
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--clip', type=float, help='gradient clipping', default='0')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='1')
args = parser.parse_args()
args.pattern_specs=OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.pattern_specs.split("_")),
                                key=lambda t: t[0]))
print(args)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import importlib
from Thiti_Model import my_utils

import os
import imp
import re


from mimic3models import common_utils
from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader
from mimic3models.preprocessing import Discretizer, Normalizer


import torch
seed=0
import random
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import numpy as np
np.random.seed(seed)
from Thiti_Model import Sopa_Decomp
from mimic3models import metrics


print('Preparing training data ... ')
train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data, 'train'), listfile=os.path.join(args.data, 'train_listfile.csv'), small_part=args.small_part)
val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data, 'train'), listfile=os.path.join(args.data, 'val_listfile.csv'), small_part=args.small_part)
test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data, 'test'), listfile=os.path.join(args.data, 'test_listfile.csv'), small_part=args.small_part)

discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = 'decompensation/decomp_normalizer'
normalizer_state = os.path.join(os.path.dirname(args.data), normalizer_state)
normalizer.load_params(normalizer_state)

train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                normalizer, args.batch_size, shuffle=True, return_names=True)
val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                            normalizer, args.batch_size, shuffle=False, return_names=True)
test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                            normalizer, args.batch_size, shuffle=False, return_names=True)

'''Model structure'''
print('Constructing model ... ')
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
print("available device: {}".format(device))


model=Sopa_Decomp.SoPa_MLP(input_dim=args.input_dim,
                 pattern_specs=args.pattern_specs,
                 semiring = Sopa_Decomp.LogSpaceMaxTimesSemiring,
                 mlp_hidden_dim=args.mlp_hidden_dim,
                 num_mlp_layers=args.num_mlp_layers,
                 num_classes=1,
                 deep_supervision= True,                                  
                 gpu=True,
                 dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip >0 :
    my_utils.enable_gradient_clipping(model, clip=args.clip)

'''Train phase'''
print('Start training ... ')

train_loss = []
val_loss = []
batch_loss = []
max_auprc = 0
file_name = './saved_weights/'+args.file_name
for each_chunk in range(args.epochs):
    cur_batch_loss = []
    model.train()
    for each_batch in range(train_data_gen.steps):
        batch_data = next(train_data_gen)
        batch_ts = batch_data['ts']
        batch_data = batch_data['data']
        
        batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
        batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
        

        optimizer.zero_grad()
        cur_output = model(batch_x.transpose(1,2),batch_mask)
        masked_output = cur_output.unsqueeze(2) * batch_mask         
        loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
        loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
        loss = torch.neg(torch.sum(loss))
        cur_batch_loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
                
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
        for each_batch in range(val_data_gen.steps):
            valid_data = next(val_data_gen)
            valid_ts = valid_data['ts']
            valid_data = valid_data['data']
            
            valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
            valid_mask = torch.tensor(valid_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
            valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)
            
            valid_output = model(valid_x.transpose(1,2),valid_mask)
            
            valid_output=valid_output.unsqueeze(2)
            masked_valid_output = valid_output * valid_mask
                    
            valid_loss = valid_y * torch.log(masked_valid_output + 1e-7) + (1 - valid_y) * torch.log(1 - masked_valid_output + 1e-7)
            valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
            valid_loss = torch.neg(torch.sum(valid_loss))
            cur_val_loss.append(valid_loss.cpu().detach().numpy())
            
            for m, t, p in zip(valid_mask.cpu().numpy().flatten(), valid_y.cpu().numpy().flatten(), valid_output.cpu().detach().numpy().flatten()):
                if np.equal(m, 1):
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
    
    for each_batch in range(test_data_gen.steps):
        test_data = next(test_data_gen)
        test_ts = test_data['ts']
        test_data = test_data['data']

        test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
        test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
        test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)
        
        test_output = model(test_x.transpose(1,2),test_mask)

                
        test_output = test_output.unsqueeze(2)
        masked_test_output = test_output * test_mask

        test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(1 - masked_test_output + 1e-7)
        test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
        test_loss = torch.neg(torch.sum(test_loss))
        cur_test_loss.append(test_loss.cpu().detach().numpy())
        
        for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
            if np.equal(m, 1):
                test_true.append(t)
                test_pred.append(p)
                
    print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
    print('\n')
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    test_ret = metrics.print_metrics_binary(test_true, test_pred)