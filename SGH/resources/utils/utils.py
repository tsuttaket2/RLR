import pandas as pd
import os, json, csv, sys
csv.field_size_limit(sys.maxsize)

#################### STAN ###########################
def pprint(string):
    print(string)
    #sys.stdout.write(string+'\n')

def write_as_csv_omit_idx(filename, rows, header, omit_idx):
    def write_csv_omit_idx(llist, omit_idx, omit_str, fout):
        for i in range(len(llist)):
            if i == omit_idx:  fout.write(str(omit_str))
            else:
                if pd.isnull(llist[i]): 
                    fout.write('')
                elif (',' in str(llist[i])) or ('|' in str(llist[i])):
                    fout.write(f'"{str(llist[i])}"')
                else:
                    fout.write(str(llist[i]))
            if i != len(llist)-1: fout.write(',')
            else:                 fout.write('\n')

    assert omit_idx == 0, f'expect omit_idx to be 0, but got {omit_idx}'
    with open(filename, 'w') as fout:
        write_csv_omit_idx(header, omit_idx, '', fout)
        for idx, r in enumerate(rows):
            for i in range(len(r)):
                if i == omit_idx: fout.write(str(idx))
                else:
                    if pd.isnull(r[i]): 
                        fout.write('')
                    elif (',' in str(r[i])) or ('|' in str(r[i])):
                        fout.write(f'"{str(r[i])}"')
                    else:
                        fout.write(str(r[i]))
                if i != len(r)-1: fout.write(',')
                else:             fout.write('\n')


def get_start_end_indices(num_items, n_proc, pid):
    assert pid < n_proc <= num_items
    chunk = int(num_items / n_proc)
    chunkplus1 = chunk + 1
    remainder = num_items % n_proc
    if pid < remainder:
        start = pid * chunkplus1
        end = start + chunkplus1
    else:
        start = remainder * chunkplus1 + (pid - remainder) * chunk
        end = start + chunk
    return start, end

def split_by_delimiter(string, delimit_char):
    return [ '{}'.format(x.strip()) for x in list(csv.reader([string.strip()], delimiter=delimit_char, quotechar='"'))[0] ]

def split_by_comma(string): return split_by_delimiter(string, ',')

def write_list_as_csv(llist, fout, check_nan=False):
    for i in range(len(llist)):
        if check_nan and pd.isnull(llist[i]): llist[i] = ''
        fout.write( str(llist[i]) )
        if i != len(llist) - 1: fout.write(',')
        else:                   fout.write('\n')

def get_header_to_idx_map(header): return dict([(h.strip(), i) for i, h in enumerate(header)])
def get_values_to_idx_map(values): return get_header_to_idx_map(values)

def get_dataframe_header_and_rows(df):
    header = df.columns.values
    rows = df.values
    header_to_idx_map = get_header_to_idx_map(header)
    return header, rows, header_to_idx_map

def read_episode_str_categorical_col2(subject_path, dtype_specs, header, header_to_idx_map):
    episode = pd.read_csv(os.path.join(subject_path, 'episode.csv'), header=0, index_col=None,
                          error_bad_lines=False, dtype=dtype_specs)
    if header is None: #do this once
        header = episode.columns.values
        header_to_idx_map = get_header_to_idx_map(header)
    rows = episode.values
    charttime_idx = header_to_idx_map['CHARTTIME']
    for r in rows:
        r[charttime_idx] = pd.to_datetime(r[charttime_idx]) #convert charttime from string to time object
    return header, rows, header_to_idx_map

def get_time_str2(prefix_str, sec):
    if sec < 60:    return '%s%f secs' % (prefix_str, sec)
    if sec < 3600:  return '%s%d mins, %d secs' % (prefix_str, int(sec/60),   sec-int(sec/60)*60)
    if sec < 86400: return '%s%d hrs, %f mins'  % (prefix_str, int(sec/3600), (sec-int(sec/3600)*3600)/60.0)
    return                 '%s%d days, %f hrs'  % (prefix_str, int(sec/86400),(sec-int(sec/86400)*86400)/3600.0)

def get_time_str(sec): return get_time_str2('', sec)
def time_str(sec):     return get_time_str2('', sec)

################################################
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col, error_bad_lines=False, dtype=str)
def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events.loc[events.VALUE.notnull()]
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    return events
def read_events_str(subject_path, remove_null=True):
    path = os.path.join(subject_path, 'events.csv')
    events = pd.read_csv(path, header=0, index_col=None,error_bad_lines=False,dtype=str)
    if remove_null:
        events = events.loc[events.VALUE.notnull()]
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    return events


def read_episode(subject_path):
    episode = dataframe_from_csv(os.path.join(subject_path, 'episode.csv'))
    episode.CHARTTIME=pd.to_datetime(episode.CHARTTIME)
    return episode
def read_mortality(subject_path):
    mortality = dataframe_from_csv(os.path.join(subject_path, 'mortality.csv'), index_col=None)
    mortality.DEATH_DATE=pd.to_datetime(mortality.DEATH_DATE)
    return mortality


def read_episode_str_categorical_col(subject_path):
    config_path=os.path.join('.', 'resources/discretizer_config.json')
    with open(config_path) as f:
        config = json.load(f)
        is_categorical_channel = config['is_categorical_channel']
    dtype_specs={k:str for k,v in is_categorical_channel.items() if v}
    episode = pd.read_csv(os.path.join(subject_path, 'episode.csv'), header=0, index_col=0,error_bad_lines=False,dtype=dtype_specs)
    episode.CHARTTIME=pd.to_datetime(episode.CHARTTIME)
    return episode

if __name__ == "__main__":
    #print(split_by_comma('aa'))
    print(split_by_comma('aa,"bb,cc",dd'))
    print(split_by_comma('aaa,"mix 30, isoph,max 40, pana",ccc'))

    def split_by_pipe(sstring):
        return [ '{}'.format(x.strip()) for x in list(csv.reader([sstring.strip()], delimiter='|', quotechar='"'))[0] ]

    print(split_by_pipe("mix 30, isoph|max 40, pana"))

