#!/usr/bin/env python
# coding: utf-8

# In[14]:


import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of sgh task',
                    default='/home/thitis/Research/Research_SGH/data/Fake_SGH_Data/210222_02_05_mortality')
parser.add_argument('--timestep', type=int, help='Time step',default=10)
parser.add_argument('--batch_size', type=int, help='Batch Size',default=25)
parser.add_argument('--save_path_folder', type=str, help='folder to save data', default='/home/thitis/Research/Research_SGH/data/Fake_SGH_Data/STROKE_HAEMORRHAGIC_PRIOR/Stroke_H_10timestep_Data/train')
parser.add_argument('--save_file_prefix', type=str, help='saved data prefix', default='Train_Data_Mortality_10Timestep_train_listfile1')
parser.add_argument('--folder_filename', type=str, help='folder file name',default='train')
parser.add_argument('--list_filepath', type=str, help='listfile name',default='/home/thitis/Research/Research_SGH/data/Fake_SGH_Data/210223_23_05_mortality/train_listfile.csv')

args = parser.parse_args()
print(args)


# In[15]:


import os
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

from resources.utils.data_reader import DecompensationReader
from resources.utils.preprocessing import Discretizer
from resources.utils.mortality_generator import BatchGen, BatchGen_Fulldata

from Thiti_Model import my_utils
from resources.utils import metrics
from resources.utils import common_utils


# In[16]:


reader = DecompensationReader(dataset_dir=os.path.join(args.data, args.folder_filename),
                                listfile=os.path.join(args.list_filepath))


# In[17]:


discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


# In[18]:


#discretizer_header = discretizer.transform(reader.read_example(0)["X"])[1].split('<,>')
#cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]


# In[19]:


normalizer=None


# In[21]:


nbatches = None


# In[25]:


data_gen = BatchGen_Fulldata(reader, discretizer,
                            normalizer, args.batch_size, nbatches, False,return_names=True)


# In[26]:


common_utils.create_directory(os.path.join(args.save_path))

def save_data(save_path,batch_data):
    np.savez_compressed(save_path,\
                        batch_x=batch_data['data'][0],\
                        batch_y=batch_data['data'][1],\
                        batch_mask=batch_data['masks'],\
                        batch_mask_T=batch_data['masks_T'])

for i in range(data_gen.steps):
    save_path=os.path.join(args.save_path_folder,args.save_file_prefix+'_step{}'.format(i))
    batch_data=next(data_gen)
    save_data(save_path,batch_data)






