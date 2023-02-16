#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, help='file name of trained model', default='trained_model')
parser.add_argument('--batch_size', type=int, help='Batch Size',default=25)
parser.add_argument('--batch_size_finetune', type=int, help='Batch Size for fine tuning',default=25)
parser.add_argument('--epochs', type=int, help='Epochs',default=100)
parser.add_argument('--epochs_reg', type=int, help='Epochs for regularization',default=20)
parser.add_argument('--min_epoch', type=int, help='Only consider the performance after this round of epoch',default=5)
parser.add_argument('--gpu', type=str, help='GPU', default='0')

parser.add_argument('--pattern_specs', type=str, help='pattern specs',
                    default='5-5_20-5_40-5')
parser.add_argument('--mlp_hidden_dim', type=int, help='mlp_hidden_dim', default='2')
parser.add_argument('--num_mlp_layers', type=int, help='num_mlp_layers', default='1')
parser.add_argument('--input_dim', type=int, help='input dimension', default='57221')
parser.add_argument('--dropout', type=float, help='dropout', default='0')
parser.add_argument('--clip', type=float, help='gradient clipping', default='0')
parser.add_argument('--lr', type=float, help='learning rate', default='0.01')

parser.add_argument('--data', type=str, help='Path to the data of sgh task',
                    default='.')
parser.add_argument('--listfile_folder', type=str, help='folder containing listfile',
                    default='.')
parser.add_argument('--train_nbatches', type=int, help='train nbatches',default=30)
parser.add_argument('--valid_nbatches', type=int, help='valid nbatches',default=20)
parser.add_argument('--train_nbatches_finetune', type=int, help='train nbatches for fine tuning',default=30)
parser.add_argument('--valid_nbatches_finetune', type=int, help='valid nbatches for fine tuning',default=20)

parser.add_argument('--reg_goal_params', type=int, help='reg_goal_params',default=20)
parser.add_argument('--distance_from_target', type=int, help='distance_from_target',default=3)
parser.add_argument('--reg_strength', type=float, help='reg_strength',default=0.01)
parser.add_argument('--output_dir', type=str, help='Path to output dir',
                    default='.')


args = parser.parse_args()

args.pattern_specs=OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.pattern_specs.split("_")),
                                key=lambda t: t[0]))
print(args)


# In[2]:



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import imp
import re
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

from resources.utils.data_reader import SGHReader
from resources.utils.preprocessing import Discretizer
from resources.utils.mortality_generator import BatchGen,BatchGen_Fulldata
from Thiti_Model import Sopa_SGH_noreducedim as Sopa_SGH 
from Thiti_Model import my_utils
from resources.utils import metrics
from Thiti_Model.Sopa_SGH_noreducedim import train_reg_str
from Thiti_Model import Sopa_SGH_noreducedim as Sopa_Decomp
from Thiti_Model.my_utils import remove_old,to_file,enable_gradient_clipping
from Thiti_Model.Sopa_SGH_noreducedim import SoPa_MLP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import log_loss

# In[3]:


train_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                    listfile=os.path.join(args.listfile_folder, 'train_listfile.csv'))
valid_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'valid_listfile.csv'))


# In[4]:


train_nbatches=args.train_nbatches
valid_nbatches=args.valid_nbatches


# In[5]:


train_data_gen = BatchGen(train_reader, args.batch_size, train_nbatches, True,return_names=True)
valid_data_gen = BatchGen(valid_reader, args.batch_size, valid_nbatches, False,return_names=True)


# In[6]:

min_epoch = args.min_epoch
mlp_hidden_dim = args.mlp_hidden_dim
num_mlp_layers = args.num_mlp_layers
reg_goal_params =args.reg_goal_params
batch_size = args.batch_size
num_epochs = args.epochs_reg
distance_from_target = args.distance_from_target
dropout=args.dropout
pattern_specs=args.pattern_specs

learning_rate= args.lr
run_scheduler = True
gpu =  True
clip = True
patience = 25
semiring = Sopa_SGH.LogSpaceMaxTimesSemiring
num_classes=1
reg_strength = args.reg_strength
deep_supervision=True


# In[7]:


logging_path = os.path.join(args.output_dir ,"Trained_Paramsgoal" , args.file_name)


# In[8]:


args.input_dim=train_reader.read_example(0)['X'].shape[1]


# In[10]:


'''Model structure'''
print('Constructing model ... ')
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
print("available device: {}".format(device))
model=SoPa_MLP(input_dim=args.input_dim,
                 pattern_specs=args.pattern_specs,
                 semiring = Sopa_SGH.LogSpaceMaxTimesSemiring,
                 mlp_hidden_dim=args.mlp_hidden_dim,
                 num_mlp_layers=args.num_mlp_layers,
                 num_classes=1,
                 deep_supervision= True,                                  
                 gpu=True,
                 dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip >0 :
    my_utils.enable_gradient_clipping(model, clip=args.clip)


# In[11]:


def search_reg_str_l1(pattern_specs, semiring, mlp_hidden_dim, num_mlp_layers,num_classes,deep_supervision,
                      dropout, reg_strength, reg_goal_params, distance_from_target,
                      train_data_gen, valid_data_gen):
    # the final number of params is within this amount of target
    smallest_reg_str = 10**-9
    largest_reg_str = 10**2
    starting_reg_str = reg_strength
    found_good_reg_str = False
    too_small = False
    too_large = False
    counter = 0
    reg_str_growth_rate = 2.0
    reduced_model_path = ""
    while not found_good_reg_str:
        # deleting models which aren't going to be used

        remove_old(reduced_model_path)

        # if more than 25 regularization strengths have been tried, throw out hparam assignment and resample
        if counter > 25:
            return (counter, "bad_hparams", dev_acc, learned_d_out, reduced_model_path)

        counter += 1
        model=SoPa_MLP(input_dim=args.input_dim,
                 pattern_specs=pattern_specs,
                 semiring = semiring,
                 mlp_hidden_dim=mlp_hidden_dim,
                 num_mlp_layers=num_mlp_layers,
                 num_classes=num_classes,
                 deep_supervision= deep_supervision,                                  
                 gpu=True,
                 dropout=dropout).to(device)
        
        dev_acc, learned_d_out, reduced_model_path = Sopa_Decomp.train_reg_str(train_data_gen, valid_data_gen, model, num_epochs, learning_rate,                                                                            
                                                                            run_scheduler, gpu, clip, patience, reg_strength, logging_path, min_epoch)
        if not(learned_d_out):
            num_params = reg_goal_params - distance_from_target-1
            print("num_params ",None)
        else: 
            num_params = sum(list(learned_d_out.values()))
            print("num_params ",num_params)
        if num_params < reg_goal_params - distance_from_target:
            if too_large:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_large = False
            too_small = True
            reg_strength = reg_strength / reg_str_growth_rate
            if reg_strength < smallest_reg_str:
                reg_strength = starting_reg_str
                return (counter, "too_small_lr", dev_acc, learned_d_out, reduced_model_path)
        elif num_params > reg_goal_params + distance_from_target:
            if too_small:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_small = False
            too_large = True
            reg_strength = reg_strength * reg_str_growth_rate

            if reg_strength > largest_reg_str:
                reg_strength = starting_reg_str
        else:
            found_good_reg_str = True
    return counter, "okay_lr", dev_acc, learned_d_out, reduced_model_path


# In[12]:


one_search_counter, lr_judgement, cur_valid_accuracy, learned_d_out, reduced_model_path=search_reg_str_l1(pattern_specs, semiring, mlp_hidden_dim, num_mlp_layers,num_classes,deep_supervision,
                      dropout, reg_strength, reg_goal_params, distance_from_target,train_data_gen, valid_data_gen)


print(f"counter : {one_search_counter}")
print(f"lr_judgement : {lr_judgement}")
print(f"dev_acc : {cur_valid_accuracy}")
print(f"learned_d_out : {learned_d_out}")
print(f"reduced_model_path : {reduced_model_path}")


print("available device: {}".format(device))
model=SoPa_MLP(input_dim=args.input_dim,
                 pattern_specs= learned_d_out,
                 semiring = Sopa_SGH.LogSpaceMaxTimesSemiring,
                 mlp_hidden_dim=args.mlp_hidden_dim,
                 num_mlp_layers=args.num_mlp_layers,
                 num_classes=1,
                 deep_supervision= True,                                  
                 gpu=True,
                 dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.clip >0 :
    my_utils.enable_gradient_clipping(model, clip=args.clip)


# In[34]:


def String_from_Ordereddict(pattern_specs):
    out=''
    for p,n in pattern_specs.items():
        out+=str(p)+'-'+str(n)+'_'
    return out[:-1]


# In[37]:


print('Start training finetuned model ... ')

train_nbatches=args.train_nbatches_finetune
valid_nbatches=args.valid_nbatches_finetune


# In[5]:


train_data_gen = BatchGen(train_reader, args.batch_size_finetune, train_nbatches, True,return_names=True)
valid_data_gen = BatchGen(valid_reader, args.batch_size_finetune, valid_nbatches, False,return_names=True)


train_loss = []
val_loss = []
batch_loss = []
max_auprc = 0
file_name = os.path.join('./saved_weights',args.file_name+'_'+String_from_Ordereddict(learned_d_out)+'.model')
for each_chunk in range(args.epochs):
    cur_batch_loss = []
    model.train()
    for each_batch in range(train_data_gen.steps):
        optimizer.zero_grad()
        
        batch_data = next(train_data_gen)
        
        batch_x = torch.tensor(batch_data['data'][0], dtype=torch.float32).to(device)
        batch_y = torch.tensor(batch_data['data'][1], dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_data['masks'], dtype=torch.float32).to(device)
        batch_mask_T = torch.tensor(batch_data['masks_T'], dtype=torch.float32).to(device)

        batch_x=batch_x.transpose(1,2)
        cur_output=model.forward(batch_x,batch_mask)
        masked_output=cur_output[batch_mask_T.nonzero(as_tuple=True)]
        
        
        loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
        loss = torch.mean(loss)
        loss = torch.neg(torch.sum(loss))
        cur_batch_loss.append(loss.cpu().detach().numpy())
        
        loss.backward()
        optimizer.step()
        
        if each_batch % 10 == 0:
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
        for each_batch in range(valid_data_gen.steps):

            batch_data = next(valid_data_gen)
            
            batch_x = torch.tensor(batch_data['data'][0], dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_data['data'][1], dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_data['masks'], dtype=torch.float32).to(device)
            batch_mask_T = torch.tensor(batch_data['masks_T'], dtype=torch.float32).to(device)

            batch_x=batch_x.transpose(1,2)
            cur_output=model.forward(batch_x,batch_mask)
            masked_output=cur_output[batch_mask_T.nonzero(as_tuple=True)]

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


# In[39]:


try :
    del(valid_data_gen)
    del(train_data_gen)
    del(train_reader)
    del(valid_reader)
except:
    print("No Deletion")
test_reader = SGHReader(dataset_dir=os.path.join(args.data),
                                  listfile=os.path.join(args.listfile_folder, 'test_listfile.csv'))

test_nbatches=None
test_data_gen = BatchGen_Fulldata(test_reader, args.batch_size_finetune, test_nbatches, False,return_names=True)

'''Evaluate phase'''
print('Testing model ... ')

checkpoint = torch.load(file_name)
save_chunk = checkpoint['chunk']
print("last saved model is in chunk {}".format(save_chunk))
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


# In[40]:


with torch.no_grad():
    cur_test_loss = []
    test_true = []
    test_pred = []
    
    for each_batch in range(test_data_gen.steps):
        batch_data = next(test_data_gen)

        batch_x = torch.tensor(batch_data['data'][0], dtype=torch.float32).to(device)
        batch_y = torch.tensor(batch_data['data'][1], dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_data['masks'], dtype=torch.float32).to(device)
        batch_mask_T = torch.tensor(batch_data['masks_T'], dtype=torch.float32).to(device)

        batch_x=batch_x.transpose(1,2)
        cur_output=model.forward(batch_x,batch_mask)
        masked_output=cur_output[batch_mask_T.nonzero(as_tuple=True)]
        
        loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
        loss = torch.mean(loss)
        loss = torch.neg(torch.sum(loss))
        cur_test_loss.append(loss.cpu().detach().numpy())
        
        for t, p in zip(batch_y.cpu().numpy().flatten(), masked_output.cpu().detach().numpy().flatten()):
            test_true.append(t)
            test_pred.append(p)
    
    print('Test neg log-likelihood = %.4f'%(log_loss(test_true, test_pred)))
    print('\n')
    test_pred = np.array(test_pred)
    test_pred = np.stack([1 - test_pred, test_pred], axis=1)
    test_ret = metrics.print_metrics_binary(test_true, test_pred)


