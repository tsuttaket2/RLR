#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import time
import argparse, os, time, json, math, gzip
from resources.utils.utils import time_str, split_by_comma, split_by_delimiter, get_values_to_idx_map
from resources.utils.utils import get_header_to_idx_map, get_start_end_indices, write_list_as_csv, pprint
from multiprocessing import Process
import os
import numpy as np
from resources.utils.common_utils import create_directory
MULTIPLE_VALUES_DELIMITER = '|'
import h5py
END_DATE_TRAILING_FLAG='_0'   #Thiti handle prescription end date
# In[2]:


class FeatureInfo:
    def __init__(self, id_to_feature, feature_to_id, is_categorical_feature, possible_values, feature_set,
                 num_out_features, num_out_features_mask, out_begin_pos, out_begin_pos_mask, is_multiple_codes, 
                 variable_impute_strategy): #Thiti
        self.id_to_feature          = id_to_feature
        self.feature_to_id          = feature_to_id
        self.is_categorical_feature = is_categorical_feature
        self.possible_values        = possible_values
        self.feature_set            = feature_set
        self.num_out_features       = num_out_features
        self.num_out_features_mask  = num_out_features_mask
        self.out_begin_pos          = out_begin_pos
        self.days_idx = -1
        self.is_multiple_codes      = is_multiple_codes  #Thiti
        self.out_begin_pos_mask     = out_begin_pos_mask
        self.variable_impute_strategy = variable_impute_strategy
        


# In[3]:


#Thiti
def compute_length_and_begin_end_pos_onlymupltiplecodes_helper(id_to_feature, is_categorical_feature, possible_values,
                                            num_cells_for_non_categorical,is_multiple_codes):
    num_features = len(id_to_feature)
    cur_len = 0
    begin_pos = [0] * num_features
    end_pos   = [0] * num_features
    for i in range(num_features):
        feature = id_to_feature[i]
        begin_pos[i] = cur_len
        if is_categorical_feature[feature] and is_multiple_codes[feature]:
            end_pos[i] = begin_pos[i] + len(possible_values[feature])
        elif is_categorical_feature[feature] and not(is_multiple_codes[feature]):
            end_pos[i] = begin_pos[i] + 1
        else:
            end_pos[i] = begin_pos[i] + num_cells_for_non_categorical
        cur_len = end_pos[i]
    return cur_len, begin_pos, end_pos
def compute_length_and_begin_end_pos_helper(id_to_feature, is_categorical_feature, possible_values,
                                            num_cells_for_non_categorical):
    num_features = len(id_to_feature)
    cur_len = 0
    begin_pos = [0] * num_features
    end_pos   = [0] * num_features
    for i in range(num_features):
        feature = id_to_feature[i]
        begin_pos[i] = cur_len
        if is_categorical_feature[feature]:
            end_pos[i] = begin_pos[i] + len(possible_values[feature])
        else:
            end_pos[i] = begin_pos[i] + num_cells_for_non_categorical
        cur_len = end_pos[i]
    return cur_len, begin_pos, end_pos

def compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature, possible_values):
    return compute_length_and_begin_end_pos_helper(id_to_feature, is_categorical_feature, possible_values, 1)


# In[4]:


def create_out_header(id_to_feature, is_categorical_feature, possible_values):
    length, begin_pos, end_pos = compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature,
                                                                             possible_values)
    assert len(begin_pos) == len(end_pos) == len(id_to_feature)

    out_header = [''] * (length)  # len(out_header) == 57265
    for i in range(len(id_to_feature)):
        feature = id_to_feature[i]
        sidx = begin_pos[i]
        eidx = end_pos[i]
        if is_categorical_feature[feature]:
            assert eidx-sidx == len(possible_values[feature])
            for value, idx in possible_values[feature].items():
                feature_name = feature+"_"+value
                if ',' in feature_name:
                    feature_name=f'"{feature_name}"'
                out_header[sidx+idx] = feature_name
        else:
            assert eidx-sidx == 1
            feature_names = [feature]
            for j in range(len(feature_names)):
                out_header[sidx+j] = feature_names[j]

    return out_header


# In[5]:


#Thiti
def create_out_header_mask(id_to_feature, is_categorical_feature,is_multiple_codes, possible_values):
    length, begin_pos, end_pos = num_out_features_mask, out_begin_pos_mask, _ = compute_length_and_begin_end_pos_onlymupltiplecodes_helper(id_to_feature, is_categorical_feature,
                                                                possible_values,1,is_multiple_codes)
    assert len(begin_pos) == len(end_pos) == len(id_to_feature)

    out_header = [''] * (length) 
    for i in range(len(id_to_feature)):
        feature = id_to_feature[i]
        sidx = begin_pos[i]
        eidx = end_pos[i]
        if is_categorical_feature[feature] and is_multiple_codes[feature]:
            assert eidx-sidx == len(possible_values[feature])
            for value, idx in possible_values[feature].items():
                feature_name = feature+"_"+value
                if ',' in feature_name:
                    feature_name=f'"{feature_name}"'
                out_header[sidx+idx] = feature_name
        elif is_categorical_feature[feature] and not is_multiple_codes[feature]:
            assert eidx-sidx == 1
            feature_names = [feature]
            for j in range(len(feature_names)):
                out_header[sidx+j] = feature_names[j]
        else:
            assert eidx-sidx == 1
            feature_names = [feature]
            for j in range(len(feature_names)):
                out_header[sidx+j] = feature_names[j]

    return out_header

def create_out_variable_imputation(id_to_feature, is_categorical_feature, possible_values, variable_impute_strategy):
    length, begin_pos, end_pos = compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature,
                                                                             possible_values)
    assert len(begin_pos) == len(end_pos) == len(id_to_feature)

    variable_imputation = [''] * (length)  # len(out_header) == 57265
    for i in range(len(id_to_feature)):
        feature = id_to_feature[i]
        sidx = begin_pos[i]
        eidx = end_pos[i]
        if variable_impute_strategy[feature]=='previous':
            for j in range(sidx,eidx):
                variable_imputation[j] = 'previous'
        elif variable_impute_strategy[feature]=='zero':
            for j in range(sidx,eidx):
                variable_imputation[j] = 'zero'

    return variable_imputation


# In[6]:


def get_csv_filename_ylabel_pairs(list_filename):
    with open(list_filename) as fin:
        lines = fin.readlines()
    lines = lines[1:] #ignore header
    file_ylabel_pairs = []
    for line in lines:
        in_subject_csv_filename, _, y_label = split_by_comma(line)
        file_ylabel_pairs.append( (in_subject_csv_filename, y_label) )
    return file_ylabel_pairs

def sample_neg_examples(file_ylabel_pairs, frac_neg_keep, pid):
    assert 0 <= frac_neg_keep <= 1.0
    num_neg = 0
    for _, ylabel in file_ylabel_pairs:
        if ylabel == '0': num_neg += 1
    num_neg_keep = int(frac_neg_keep * num_neg)

    cnt = 0
    last_idx = len(file_ylabel_pairs)
    for i, (_, ylabel) in enumerate(file_ylabel_pairs):
        if ylabel == '0':
            cnt += 1
            if cnt >= num_neg_keep:
                last_idx = i
                break
    pprint(f'{pid}: keep {cnt} out of {num_neg} examples; {last_idx} examples (+ve & -ve)  kept')
    return file_ylabel_pairs[:last_idx]


# In[7]:


def populate_feature2(out_features, out_features_mask, out_features_prev, feature, value, feature_info):
    feature_to_id          = feature_info.feature_to_id
    is_categorical_feature = feature_info.is_categorical_feature
    possible_values        = feature_info.possible_values
    begin_pos              = feature_info.out_begin_pos
    is_multiple_codes      = feature_info.is_multiple_codes   #Thiti
    begin_pos_mask         = feature_info.out_begin_pos_mask
    #Thiti handle prescription end date
    def get_category_ids():
        value_to_id_map = possible_values[feature]
        cat_id = value_to_id_map.get(value)
        if cat_id is not None: return [cat_id] ,[]             #Thiti handle prescription end date
        values = split_by_delimiter(value, MULTIPLE_VALUES_DELIMITER)
        cat_ids = []      #categories to set to 1
        cat_ids_zero = [] #categories to set to 0
        for i,v in enumerate(values):
            cid = value_to_id_map.get(v)
            if cid is not None:
                cat_ids.append(cid)
            else: # this is categorical value that must be set to 0
                idx = v.rfind(END_DATE_TRAILING_FLAG)
                assert idx >= 0, f'bad feature value {v}'
                #Debug
                original_v=v
                v = v[:idx]
                cid = value_to_id_map.get(v)
                assert cid is not None, f'value {v} is not found for {feature}'
                cat_ids_zero.append(cid)
        return cat_ids, cat_ids_zero

    def convert_to_float(sstr):
        try:
            v = float(sstr)
            return v
        except ValueError: #handle multiple values for numerical feature
            str_list = split_by_delimiter(sstr, MULTIPLE_VALUES_DELIMITER)
            total = 0.0
            for s in str_list:
                total += float(s)
            v = total / len(str_list)
            return v
    feature_id = feature_to_id[feature]
    if is_categorical_feature[feature] and is_multiple_codes[feature]:
        num_values  = len(possible_values[feature])
        one_hot     = np.zeros((num_values,))
        category_ids, category_ids_zero = get_category_ids()        #Thiti handle prescription end date
        
        s_idx = begin_pos[feature_id]
        e_idx = s_idx + num_values
        s_idx_mask = begin_pos_mask[feature_id]
        e_idx_mask = s_idx_mask + num_values
        
        for cid in category_ids:
            out_features[s_idx+cid]=1
            out_features_mask[s_idx_mask+cid] = 1
            out_features_prev[s_idx+cid]= 1
        for cid in category_ids_zero:                            #Thiti handle prescription end date
            out_features[s_idx+cid]=0
            out_features_mask[s_idx_mask+cid] = 1
            out_features_prev[s_idx+cid]= 0

    elif is_categorical_feature[feature] and not is_multiple_codes[feature]:
        num_values  = len(possible_values[feature])
        one_hot     = np.zeros((num_values,))
        category_ids, category_ids_zero = get_category_ids()        #Thiti handle prescription end date
        
        s_idx = begin_pos[feature_id]
        e_idx = s_idx + num_values
        for cid in category_ids: 
            one_hot[cid] = 1
            out_features[s_idx+cid]=1
        for cid in category_ids_zero:                            #Thiti handle prescription end date
            one_hot[cid] = 0
            out_features[s_idx+cid]=0

        #To prevent multiple 1's
        out_features_prev[s_idx:e_idx]=one_hot
        out_features_mask[begin_pos_mask[feature_id]]=1
    else:
        out_features[begin_pos[feature_id]] = str(convert_to_float(value))
        out_features_mask[begin_pos_mask[feature_id]]=1
        out_features_prev[begin_pos[feature_id]] = str(convert_to_float(value))


# In[8]:


def create_features2(in_feature_list_bin, in_header, feature_info, out_features_prev):
    feature_set           = feature_info.feature_set
    num_out_features      = feature_info.num_out_features
    num_out_features_mask = feature_info.num_out_features_mask
    out_features = np.array([''] * num_out_features,dtype=np.dtype(object))
    out_features_mask = np.array([0] * num_out_features_mask,dtype=np.dtype(int))
    #Time
    #start_sec_create_features = time.perf_counter()
    for j in range(len(in_feature_list_bin)):     
        for i in range(len(in_feature_list_bin[j])):
            feature = in_header[i]
            value   = in_feature_list_bin[j][i]
            if value == '' or feature not in feature_set: continue
            #start_sec_pop_feat = time.perf_counter()
            populate_feature2(out_features, out_features_mask, out_features_prev, feature, value, feature_info)
            #print(f'populate_feature2 took {time_str(time.perf_counter()-start_sec_pop_feat)}')

    #print(f'create_features2 took {time_str(time.perf_counter()-start_sec_create_features)}')
    return out_features, out_features_mask


# In[9]:


def process_subject_csv_file2(csv_filename, fout, feature_info, max_past_years,timestep):
    eps = 1e-6
    def process_line(line, in_header):
        line = line.strip()
        in_features = split_by_comma(line)
        out_features = create_features(in_features, in_header, feature_info)
        return out_features
    if max_past_years > 0:
        with open(csv_filename) as fin:
            lines = fin.readlines()
        in_header = split_by_comma(lines[0])
        if feature_info.days_idx < 0:
            header_to_index_map = get_header_to_idx_map(in_header)
            feature_info.days_idx = header_to_index_map['DAYS']
        days_idx = feature_info.days_idx
        last_line = lines[-1]
        last_day = int(split_by_comma(last_line)[days_idx])
        earliest_day = int(last_day) - max_past_years*365

        first_idx = 0
        lines = lines[1:]
        for i in range(len(lines)):
            line = lines[i]
            day = split_by_comma(line)[days_idx]
            if int(day) >= earliest_day:
                first_idx = i
                break
        lines = lines[first_idx:]
        earliest_day = int(split_by_comma(lines[0])[days_idx])
        N_bins = int((last_day-earliest_day) / timestep + 1.0 - eps)
        lines = np.array(lines[1:])
        X = np.full((N_bins,feature_info.num_out_features), '', dtype=object)
        X_mask = np.full((N_bins,feature_info.num_out_features_mask), 0, dtype=int)
        out_features_prev = np.full(feature_info.num_out_features, '', dtype=object)
        out_variable_imputation = create_out_variable_imputation(feature_info.id_to_feature, feature_info.is_categorical_feature, 
                                                                 feature_info.possible_values, feature_info.variable_impute_strategy)
        #Imputation by previous
        imputation_check1=np.array(out_variable_imputation)=='previous'
        
        days = [int(split_by_comma(line)[days_idx]) for line in lines]
        bin_id = np.array([int((t-earliest_day) / timestep - eps) for t in days])
        unique_bin_id = list(set(bin_id))
        for t in range(N_bins):
            if t in unique_bin_id:
                lines_bin=lines[bin_id==t,]
                lines_bin=[split_by_comma(line) for line in lines_bin]
                X[t],X_mask[t] = create_features2(lines_bin, in_header, feature_info, out_features_prev)          
            imputation_check2=out_features_prev!=''
            imputation_check=np.logical_and(imputation_check1, imputation_check2)
            X[t][imputation_check]=out_features_prev[imputation_check]
        
        #Change value '' to 0
        X[X=='']=0
        X_mask[X_mask=='']=0
        X=X.astype('float32',casting='unsafe')
        X_mask=X_mask.astype('float32',casting='unsafe')
        
        fout.create_dataset('data', compression='gzip', data=X)
        fout.create_dataset('mask', compression='gzip', data=X_mask)

    else:
        with open(csv_filename) as fin:
            lines = fin.readlines()
        in_header = split_by_comma(lines[0])
        if feature_info.days_idx < 0:
            header_to_index_map = get_header_to_idx_map(in_header)
            feature_info.days_idx = header_to_index_map['DAYS']
        days_idx = feature_info.days_idx
        last_line = lines[-1]
        last_day = int(split_by_comma(last_line)[days_idx])
        
        N_bins = int(last_day / timestep + 1.0 - eps)
        first_idx = 0
        lines = np.array(lines[1:])
        X = np.full((N_bins,feature_info.num_out_features), '', dtype=object)
        X_mask = np.full((N_bins,feature_info.num_out_features_mask), 0, dtype=int)
        out_features_prev = np.full(feature_info.num_out_features, '', dtype=object)
        out_variable_imputation = create_out_variable_imputation(feature_info.id_to_feature, feature_info.is_categorical_feature, 
                                                                 feature_info.possible_values, feature_info.variable_impute_strategy)
        #Imputation by previous
        imputation_check1=np.array(out_variable_imputation)=='previous'
        
        days = [int(split_by_comma(line)[days_idx]) for line in lines]
        bin_id = np.array([int(t / timestep - eps) for t in days])
        unique_bin_id = list(set(bin_id))
        for t in range(N_bins):
            if t in unique_bin_id:
                lines_bin=lines[bin_id==t,]
                lines_bin=[split_by_comma(line) for line in lines_bin]
                X[t],X_mask[t] = create_features2(lines_bin, in_header, feature_info,out_features_prev)
                
            imputation_check2=out_features_prev!=''
            imputation_check=np.logical_and(imputation_check1, imputation_check2)
            X[t][imputation_check]=out_features_prev[imputation_check]
        
        #Change value '' to 0
        X[X=='']=0
        X_mask[X_mask=='']=0
        X=X.astype('float32',casting='unsafe')
        X_mask=X_mask.astype('float32',casting='unsafe')
        
        fout.create_dataset('data', compression='gzip', data=X)
        fout.create_dataset('mask', compression='gzip', data=X_mask)



# In[10]:


def prepare_data(proc_id, num_proc, in_list_filepath, in_csv_dir, saved_filename, config_path,
                 max_past_years, max_csv_files, gzip_files, frac_neg_examples, timestep):
    start_sec = time.perf_counter()

    with open(config_path) as fin:
        config = json.load(fin)
    id_to_feature          = config['id_to_channel']
    is_categorical_feature = config['is_categorical_channel']
    possible_values        = config['possible_values']
    is_multiple_codes        = config['is_multiple_codes']  #Thiti
    variable_impute_strategy = config['Imputation_Strategy']
    feature_set            = set(id_to_feature)
    possible_values = dict([(feature, get_values_to_idx_map(value_list)) for feature, value_list in possible_values.items()])
    feature_to_id   = dict(zip(id_to_feature, range(len(id_to_feature))))

    #compute these once here rather than repeatedly later
    #num of out features after representing categorical feature as one-hot
    ssec = time.perf_counter()
    num_out_features, out_begin_pos, _ = compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature,
                                                                          possible_values)
    num_out_features_mask, out_begin_pos_mask, _ = compute_length_and_begin_end_pos_onlymupltiplecodes_helper(id_to_feature, is_categorical_feature,
                                                                possible_values,1,is_multiple_codes)
    pprint(f'{proc_id}: compute_length_and_begin_end_pos took {time_str(time.perf_counter() - ssec)}')

    feature_info = FeatureInfo(id_to_feature, feature_to_id, is_categorical_feature, possible_values, feature_set,
                            num_out_features,num_out_features_mask, out_begin_pos, out_begin_pos_mask,is_multiple_codes, variable_impute_strategy)

    file_ylabel_pairs = get_csv_filename_ylabel_pairs(in_list_filepath)
    file_ylabel_pairs.sort(key=lambda e: e[0])
    if 0 <= frac_neg_examples < 1.0:
        file_ylabel_pairs = sample_neg_examples(file_ylabel_pairs, frac_neg_examples, proc_id)

    sidx, eidx = get_start_end_indices(len(file_ylabel_pairs), num_proc, proc_id)
    sub_file_ylabel_pairs = file_ylabel_pairs[sidx:eidx]
    pprint(f'proc {proc_id}: processing sub_file_ylabel_pairs[{sidx}:{eidx}]')
    if 0 <= max_csv_files < len(sub_file_ylabel_pairs):
        sub_file_ylabel_pairs = sub_file_ylabel_pairs[:max_csv_files]
        pprint(f'proc {proc_id}: limit to {len(sub_file_ylabel_pairs)} csv files')


    ssec = time.perf_counter()
    out_header = create_out_header(id_to_feature, is_categorical_feature, possible_values)
    out_header_mask=create_out_header_mask(id_to_feature, is_categorical_feature, is_multiple_codes, possible_values)
    out_variable_imputation = create_out_variable_imputation(id_to_feature, is_categorical_feature, possible_values, variable_impute_strategy)
    pprint(f'{proc_id}: create_out_header took {time_str(time.perf_counter() - ssec)}')

    ssec = time.perf_counter()
    cnt = 0
    for in_subject_csv_filename, y_label in sub_file_ylabel_pairs:
        saved_filename_subj=os.path.join(saved_filename,in_subject_csv_filename)
        
        fout = h5py.File(saved_filename_subj, 'w')
        
            
        if cnt % 1000 == 0:
            pprint(f'{proc_id}: processing {cnt} csv files took {time_str(time.perf_counter() - ssec)}')

        out_header=create_out_header(id_to_feature, is_categorical_feature, possible_values)
        out_header_mask=create_out_header_mask(id_to_feature, is_categorical_feature, is_multiple_codes, possible_values)
        out_variable_imputation = create_out_variable_imputation(id_to_feature, is_categorical_feature, possible_values, variable_impute_strategy)
        
        in_csv_fn = os.path.join(in_csv_dir, in_subject_csv_filename)
        process_subject_csv_file2(in_csv_fn, fout, feature_info, max_past_years,timestep)
        cnt += 1

        fout.create_dataset('out_header', compression='gzip', data=np.array(out_header).astype('S'))
        fout.create_dataset('out_header_mask', compression='gzip', data=np.array(out_header_mask).astype('S'))
        fout.create_dataset('timestep', compression='gzip', data=[timestep])
        fout.close()

    pprint(f'proc {proc_id}: processing csv files took {time_str(time.perf_counter() - start_sec)}')


# In[ ]:


def multiprocess(num_processes, list_filepath, in_csv_dir, saved_filename, config_path, max_past_years, max_csv_files,
                 gzip_files, frac_neg_examples, timestep):
    jobs = []
    for pid in range(_num_processes):
        p = Process(target=prepare_data,
                    args=(pid, num_processes, list_filepath, in_csv_dir, saved_filename, config_path,
                          max_past_years, max_csv_files, gzip_files, frac_neg_examples, timestep))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()


# In[11]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_csv_dir',      type=str, help='directory containing all subject.csv files', default='.')
    parser.add_argument('--list_filepath',        type=str, help='listfile name, e.g., /home/skok/train_listfile.csv',default='.')
    parser.add_argument('--saved_filename',       type=str, help='saved data file name, e.g., /home/skok/static_foot.train')
    parser.add_argument('--discretizer_filename', type=str, help='full path to discretizer json, e.g., /home/skok/discretizer_config.json')
    parser.add_argument('--max_past_years',       type=int, help='look back this number of years at most; negative value means use all data', default=-1)
    parser.add_argument('--max_csv_files',        type=int, help='max number of csv files to process; negative value means process all', default=-1)
    parser.add_argument('--number_processes', '-p', type=int, help='', default=10)
    parser.add_argument('--gzip_files',             type=int, help='0/1 gzip output files', default=1)
    parser.add_argument('--frac_neg_examples',    type=float, help='fraction of negative examples to use; negative means use all', default=-1)
    parser.add_argument('--timestep',             type=int, help='timestep', default=10)
    args = parser.parse_args()
    pprint(f'{args}')

    #example values
    #--data='output_from_mortality_folder'
    #--list_filepath='./mortality/test_listfile-1.csv'
    #--saved_filename='test_listfile-1'
    #--read_chunk_size=100
    #--n_examples=10000
    #--saved_folder='./Test_Mortality_Static_Data/' &

    # #for debugging
    # dirr='/Users/stanleykok/PycharmProjects/empower-dataprep-thiti'
    # args.subject_csv_dir = dirr+'/mortality_test/test'
    # args.list_filepath   = dirr+'/mortality_test/test_listfile.csv'
    # args.saved_filename  = 'test_listfile-1'
    # args.resources_folder= dirr+'/resources/discretizer_config.json'
    # args.max_past_years  = 5
    # args.max_csv_files   = 10
    # args.num_processes   = 2
    # args.gzip_files      = 0
    # args.frac_neg_examples = -1

    _in_csv_dir      = args.subject_csv_dir
    _list_filepath   = args.list_filepath
    _saved_filename  = args.saved_filename
    _config_path     = args.discretizer_filename #'/dir/discretizer_config.json'
    _max_past_years  = args.max_past_years
    _max_csv_files   = args.max_csv_files
    _num_processes   = args.number_processes
    _gzip_files      = bool(args.gzip_files)
    _frac_neg_examples = args.frac_neg_examples
    _timestep        = args.timestep

    begin_sec = time.perf_counter()

    pprint('multiprocessing csv files')
    start_sec3 = time.perf_counter()
    create_directory(_saved_filename)
    multiprocess(_num_processes, _list_filepath, _in_csv_dir, _saved_filename, _config_path, _max_past_years,
                 _max_csv_files, _gzip_files, _frac_neg_examples, _timestep)
    pprint(f'multiprocessing ALL csv files took {time_str(time.perf_counter() - start_sec3)}')

