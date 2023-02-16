import random
import numpy as np
import os
import json

def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)
    
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def exclusion_categorical_col(data,listfile_folder,resources_folder):
    fn=os.path.join(resources_folder, 'discretizer_config.json')
    fn=open(fn)
    json_file_content = json.load(fn)
    fn.close()

    from resources.utils.data_reader import SGHReader
    full_data_reader = SGHReader(dataset_dir=os.path.join(data),
                                  listfile=os.path.join(listfile_folder, 'valid_listfile.csv'))
    chunk_sample_data = read_chunk(full_data_reader,1)
    cnt=0
    continuous_col=[]
    all_col=set([i for i in range(len(chunk_sample_data['header'].tolist()))])
    for index, col in enumerate(chunk_sample_data['header'].tolist()):
        col=col.decode('UTF-8')
        try:
            if not json_file_content['is_categorical_channel'][col]:
                cnt+=1
                continuous_col.append(index)
        except:
            continue
    return list(all_col-set(continuous_col))

def get_col_from_feature_rank_file(feature_rank_fn,resources_folder='./resources'):
    fin = open(feature_rank_fn,'rt')
    #Get dictionary with stat for continuous var {cont_var:cont_var_MAX,...}
    config_file=os.path.join(resources_folder, 'discretizer_config.json')
    
    def cont_var_with_stat(config_file):
        with open(config_file) as fin:
            config = json.load(fin)
        cont_var_with_stat={}
        for k,v in config['is_categorical_channel'].items():
            if not v:
                k_stat = {k+suffix:k for suffix in ["_MEAN","_STD","_MIN","_MAX"]}
                cont_var_with_stat.update(k_stat)
        return cont_var_with_stat

    cont_var_with_stat_dict=cont_var_with_stat(config_file)
    #Iterating through each line of feature ranking file (of xgboost) and get the variable name
    selected_col=[]
    line = fin.readline()
    cnt = 1
    while line:
        line_split=line.split(',', 1)
        assert line_split[1][-1:]=='\n', "Not ending with newline"
        line_split[1]=line_split[1][:-1]
        try:
            col=cont_var_with_stat_dict[line_split[1]]
            selected_col.append(col)
        except:
            col=line_split[1]
            selected_col.append(col)
        line = fin.readline()
        cnt += 1
    fin.close()
    return selected_col

def exclusion_col_given_list(data,listfile_folder,resources_folder,selected_col):
    fn=os.path.join(resources_folder, 'discretizer_config.json')
    fn=open(fn)
    json_file_content = json.load(fn)
    fn.close()

    from resources.utils.data_reader import SGHReader
    full_data_reader = SGHReader(dataset_dir=os.path.join(data),
                                  listfile=os.path.join(listfile_folder, 'valid_listfile.csv'))
    chunk_sample_data = read_chunk(full_data_reader,1)
    cnt=0
    selected_col_index=[]
    all_col=set([i for i in range(len(chunk_sample_data['header'].tolist()))])
    for index, col in enumerate(chunk_sample_data['header'].tolist()):
        col=col.decode('UTF-8')
        if col in selected_col:
            selected_col_index.append(index)
            print("found")
    return list(all_col-set(selected_col_index))