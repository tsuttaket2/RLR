import argparse
from collections import OrderedDict
from Thiti_Model.my_utils import add_common_arguments,str2bool

parser = argparse.ArgumentParser()
add_common_arguments(parser)
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default='/home/thitis/Research/Research_SGH/data/mimic3benchmark/data/decompensation')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--file_name', type=str, help='file_name for model',
                    default='trained_model')
parser.add_argument('--pattern_specs', type=str, help='pattern specs',
                    default='5-5_20-5_40-5')
parser.add_argument('--mlp_hidden_dim', type=int, help='mlp_hidden_dim', default='128')
parser.add_argument('--num_mlp_layers', type=int, help='num_mlp_layers', default='1')
parser.add_argument('--input_dim', type=int, help='input dimension', default='76')
parser.add_argument('--finetuned_num_epochs', type=int, help='fine tuned num epochs', default='100')
parser.add_argument('--clip', type=str2bool, help='gradient clipping', default='True')
parser.add_argument('--gpu', type=str, help='Choose GPU', default='0')
parser.add_argument('--reg_goal_params', type=float, default=40)
parser.add_argument('--distance_from_target', type=float, default=10)

args = parser.parse_args()
args.pattern_specs=OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.pattern_specs.split("_")),
                                key=lambda t: t[0]))
print(args)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import importlib
from Thiti_Model import my_utils
from Thiti_Model.my_utils import remove_old,to_file,enable_gradient_clipping
import imp
import re


from mimic3models import common_utils
from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader
from mimic3models.preprocessing import Discretizer, Normalizer

seed=0
import torch
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


mlp_hidden_dim = args.mlp_hidden_dim
num_mlp_layers = args.num_mlp_layers
reg_goal_params =args.reg_goal_params
batch_size = args.batch_size
num_epochs = args.epochs
finetuned_num_epochs= args.finetuned_num_epochs
distance_from_target = args.distance_from_target
dropout=args.dropout
pattern_specs=args.pattern_specs

learning_rate= 0.01
run_scheduler = True
gpu =  True
clip = True
patience = 25
semiring = Sopa_Decomp.LogSpaceMaxTimesSemiring
num_classes=1
reg_strength = 0.1
deep_supervision=True


logging_path = args.output_dir +"/Trained_Paramsgoal/" + args.file_name
finetuned_logging_path=args.output_dir + "/Fine_Tuned/" + args.file_name

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
                 pattern_specs=pattern_specs,
                 semiring = Sopa_Decomp.LogSpaceMaxTimesSemiring,
                 mlp_hidden_dim=mlp_hidden_dim,
                 num_mlp_layers=num_mlp_layers,
                 num_classes=1,
                 deep_supervision= True,                                  
                 gpu=True,
                 dropout=dropout).to(device)

def search_reg_str_l1(pattern_specs, semiring, mlp_hidden_dim, num_mlp_layers,num_classes,deep_supervision,
                      dropout, reg_strength, reg_goal_params, distance_from_target):
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
        model=Sopa_Decomp.SoPa_MLP(input_dim=args.input_dim,
                 pattern_specs=pattern_specs,
                 semiring = semiring,
                 mlp_hidden_dim=mlp_hidden_dim,
                 num_mlp_layers=num_mlp_layers,
                 num_classes=num_classes,
                 deep_supervision= deep_supervision,                                  
                 gpu=True,
                 dropout=dropout).to(device)
        
        dev_acc, learned_d_out, reduced_model_path = Sopa_Decomp.train_reg_str(train_data_gen, val_data_gen, 
                                                                                model, num_epochs, learning_rate, 
                                                                                run_scheduler, gpu, clip, patience, 
                                                                                reg_strength, logging_path)
        if not(learned_d_out):
            num_params = reg_goal_params + distance_from_target+1
        else: 
            num_params = sum(list(learned_d_out.values()))

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

one_search_counter, lr_judgement, cur_valid_accuracy, learned_d_out, reduced_model_path=\
search_reg_str_l1(pattern_specs, semiring, mlp_hidden_dim, num_mlp_layers,num_classes,deep_supervision,
                      dropout, reg_strength, reg_goal_params, distance_from_target)

## Train and Fine Tune
model=Sopa_Decomp.SoPa_MLP(input_dim = args.input_dim, 
                pattern_specs = learned_d_out, 
                semiring = model.sopa.semiring, 
                mlp_hidden_dim = model.mlp_hidden_dim, 
                num_mlp_layers = model.num_mlp_layers, 
                num_classes = model.num_classes,
                deep_supervision = model.deep_supervision, 
                gpu = True, 
                dropout = model.dropout).to(device)

finetuned_acc,_,filename=Sopa_Decomp.train_no_reg(train_data_gen, val_data_gen, model, finetuned_num_epochs, 
            learning_rate, run_scheduler, gpu, clip, patience, finetuned_logging_path)

print("*"*100)
print("Fine Tuned Accuracy: ",finetuned_acc)
print("filename: ",filename)
print("*"*100)