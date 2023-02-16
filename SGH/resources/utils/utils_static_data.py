import json
import numpy as np
from scipy.stats import skew
from resources.utils import common_utils
import os
#all_functions = [min, max, np.mean, np.std, skew, len]
all_functions = [np.mean]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

#sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
#               (3, 10), (3, 25), (3, 50)]
sub_periods = [(2, 100)]

config_path=os.path.join('.', 'resources/discretizer_config.json')
with open(config_path) as f:
    config = json.load(f)
    id_to_channel = config['id_to_channel']
    channel_to_id = dict(zip(id_to_channel, range(len(id_to_channel))))
    is_categorical_channel = config['is_categorical_channel']
    possible_values = config['possible_values']
    header = config['header']

def readdata_for_nonsequential_ml(X,id_to_channel,channel_to_id,is_categorical_channel,
                                  possible_values,header):
    N_channels = len(id_to_channel)

    cur_len = 0
    begin_pos = [0 for i in range(N_channels)]
    end_pos = [0 for i in range(N_channels)]
    for i in range(N_channels):
        channel = id_to_channel[i]
        begin_pos[i] = cur_len
        if is_categorical_channel[channel]:
            end_pos[i] = begin_pos[i] + len(possible_values[channel])
        else:
            end_pos[i] = begin_pos[i] + 1
        cur_len = end_pos[i]
        

    N_bins=X.shape[0]
    data = [["" for j in range(cur_len)] for i in range(N_bins)]
        
    def write(data, bin_id, channel, value, begin_pos):
        channel_id = channel_to_id[channel]
        if is_categorical_channel[channel]:
            category_id = possible_values[channel].index(value)   
            N_values = len(possible_values[channel])
            one_hot = np.zeros((N_values,))
            one_hot[category_id] = 1

            for pos in range(N_values):
                data[bin_id][begin_pos[channel_id] + pos] = one_hot[pos]
        else:
            data[bin_id][begin_pos[channel_id]] = float(value)

    for bin_id,row in enumerate(X):
        for j in range(0, len(row)):
            if row[j] == "":
                continue

            try:
                channel = header[j]
            except:
                continue   
            if channel not in id_to_channel:
                continue

            channel_id = channel_to_id[channel]

            write(data, bin_id, channel, row[j], begin_pos)
        data[bin_id].insert(0,float(row[-1]))
    return np.array(data)

def convert_to_dict(data):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(len(data[0]) - 1)]
    for i in range(1, len(data[0])):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret

def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)
    
    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)


def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    return np.array([extract_features_single_episode(x, period, functions)
                     for x in data_raw])






def extract_features_from_rawdata(chunk, header, period, features):
    config_path=os.path.join('.', 'resources/discretizer_config.json')
    with open(config_path) as f:
        config = json.load(f)
        id_to_channel = config['id_to_channel']
        channel_to_id = dict(zip(id_to_channel, range(len(id_to_channel))))
        is_categorical_channel = config['is_categorical_channel']
        possible_values = config['possible_values']
        header = config['header']   
    data = [convert_to_dict(readdata_for_nonsequential_ml(X,id_to_channel,channel_to_id,is_categorical_channel,
                                      possible_values,header)) for X in chunk]
    return extract_features(data, period, features)

def read_and_extract_features(reader, count, period, features,read_chunk_size=10):
    Xs = []
    ys = []
    names = []
    ts = []
    for i in range(0, count, read_chunk_size):
        j = min(count, i + read_chunk_size)
        ret = common_utils.read_chunk(reader, j - i)
        X = extract_features_from_rawdata(ret['X'], ret['header'], period, features)
        Xs.append(X)
        ys += ret['y']
        names += ret['name']
        ts += ret['t']

        if i%10 ==0:
            print('Chunk %d out of %d '%(i, count))
    Xs = np.concatenate(Xs, axis=0)
    return (Xs, ys, names, ts)

def save_results(names, ts, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))
