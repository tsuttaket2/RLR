#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import argparse
import numpy as np
import resources.utils.common_utils as common_utils


# In[ ]:


parser = argparse.ArgumentParser(description='Extract per-subject data from SGH CSV files.')
parser.add_argument('--source_listfile_folder','-s', type=str, help='Directory containing source listfile.')
parser.add_argument('--target_listfile_folder','-t', type=str, help='Directory containing target listfile.')
parser.add_argument('--output_listfile_folder','-o', type=str, help='Directory to write output listfile')
args = parser.parse_args()


# In[ ]:


common_utils.create_directory(args.output_listfile_folder)

source_listfile_folder=args.source_listfile_folder
target_listfile_folder=args.target_listfile_folder
output_listfile_folder=args.output_listfile_folder
# In[3]:


src_train_dcofid=[]
src_valid_dcofid=[]
src_test_dcofid=[]
src_dcofid=[]
src_train_lines=[]
src_valid_lines=[]
src_test_lines=[]
for f in os.listdir(source_listfile_folder):
    fn=os.path.join(source_listfile_folder,f)
    if f.find("listfile")==-1: continue
    with open(fn, "r") as f1:
        lines=f1.readlines()
        header=lines[0]
        lines=lines[1:]
        if f.find('train') !=-1: src_train_lines=lines
        if f.find('valid') !=-1: src_valid_lines=lines
        if f.find('test') !=-1: src_test_lines=lines 
        for i,line in enumerate(lines):
            if f.find('train') !=-1:
                src_train_dcofid.append(line.split(',')[0])
            elif f.find('valid') !=-1:
                src_valid_dcofid.append(line.split(',')[0])
            elif f.find('test') !=-1:
                src_test_dcofid.append(line.split(',')[0])
            else:
                assert False, print("No split found")
src_dcofid=src_train_dcofid+src_test_dcofid+src_valid_dcofid   


# In[4]:


target_train_dcofid=[]
target_valid_dcofid=[]
target_test_dcofid=[]
target_dcofid=[]
target_train_lines=[]
target_valid_lines=[]
target_test_lines=[]
for f in os.listdir(target_listfile_folder):
    if f.find("listfile")==-1: continue
    fn=os.path.join(target_listfile_folder,f)
    
    with open(fn, "r") as f1:
        lines=f1.readlines()
        header=lines[0]
        lines=lines[1:]
        if f.find('train') !=-1: target_train_lines=lines
        if f.find('valid') !=-1: target_valid_lines=lines
        if f.find('test') !=-1: target_test_lines=lines 
        for i,line in enumerate(lines):
            if f.find('train') !=-1:
                target_train_dcofid.append(line.split(',')[0])
            elif f.find('valid') !=-1:
                target_valid_dcofid.append(line.split(',')[0])
            elif f.find('test') !=-1:
                target_test_dcofid.append(line.split(',')[0])
            else:
                assert False, print("No split found")
target_dcofid=target_train_dcofid+target_test_dcofid+target_valid_dcofid   


# In[5]:


intersection_dcofid=list(set(target_dcofid) & set(src_dcofid))


# In[6]:


def filter_lines(lines,dcofids,intersection_dcofid):
    test=np.isin(np.array(dcofids),intersection_dcofid)
    return list(np.array(lines)[test]), list(np.array(dcofids)[test])


# In[7]:


src_train_lines,src_train_dcofid = filter_lines(src_train_lines,src_train_dcofid,intersection_dcofid)
src_valid_lines,src_valid_dcofid = filter_lines(src_valid_lines,src_valid_dcofid,intersection_dcofid)
src_test_lines,src_test_dcofid = filter_lines(src_test_lines,src_test_dcofid,intersection_dcofid)

target_train_lines,target_train_dcofid = filter_lines(target_train_lines,target_train_dcofid,intersection_dcofid)
target_valid_lines,target_valid_dcofid = filter_lines(target_valid_lines,target_valid_dcofid,intersection_dcofid)
target_test_lines,target_test_dcofid = filter_lines(target_test_lines,target_test_dcofid,intersection_dcofid)


# In[8]:


target_lines  = target_train_lines+target_valid_lines+target_test_lines


# In[10]:


target_dcofid = target_train_dcofid+target_valid_dcofid+target_test_dcofid


# In[18]:


target_train_lines,target_train_dcofid = filter_lines(target_lines,target_dcofid,src_train_dcofid)
target_valid_lines,target_valid_dcofid = filter_lines(target_lines,target_dcofid,src_valid_dcofid)
target_test_lines,target_test_dcofid = filter_lines(target_lines,target_dcofid,src_test_dcofid)


# In[ ]:


for lines in [src_test_lines,src_valid_lines,src_train_lines]:
    if lines ==src_test_lines:
        f=open(os.path.join(output_listfile_folder,"src_test_listfile.csv"), "w")
    elif lines ==src_valid_lines:
        f=open(os.path.join(output_listfile_folder,"src_valid_listfile.csv"), "w")
    elif lines ==src_train_lines:
        f=open(os.path.join(output_listfile_folder,"src_train_listfile.csv"), "w")
    f.write(header)
    for line in lines:
        f.write(line)
f.close()


# In[20]:


for lines in [target_test_lines,target_valid_lines,target_train_lines]:
    if lines ==target_test_lines:
        f=open(os.path.join(output_listfile_folder,"target_test_listfile.csv"), "w")
    elif lines ==target_valid_lines:
        f=open(os.path.join(output_listfile_folder,"target_valid_listfile.csv"), "w")
    elif lines ==target_train_lines:
        f=open(os.path.join(output_listfile_folder,"target_train_listfile.csv"), "w")
    f.write(header)
    for line in lines:
        f.write(line)
    f.close()


# In[ ]:




